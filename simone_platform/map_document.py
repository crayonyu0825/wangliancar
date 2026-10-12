"""Obtain the exact SDK-selected OpenDRIVE document without changing SDK files.

Native I/O is restricted to the verified 3.0.0001 Map ABI. A cached filename
alone is never map identity. No URL, token or full document is published/logged.
Downloads occur at map initialization, never in the per-frame identity check.
"""
import ctypes
import glob
import hashlib
import os
import re
import time
import urllib.request
import urllib.parse

MAX_DOCUMENT_BYTES = 16*1024*1024
MAX_CACHE_FILES = 128


def _text(value):
    if isinstance(value,bytes):
        return value.split(b"\0",1)[0].decode("utf-8","strict")
    return str(value)


def _native_identity(service,structs,version):
    if version!="3.0.0001" or not hasattr(service,"SoGetHDMapData"):
        raise ValueError("MAP_IDENTITY_API_OR_ABI_UNVERIFIED")
    native_type = getattr(structs,"SimOne_Data_Map",None)
    fields = getattr(native_type,"_fields_",None)
    expected = (("openDrive",128,0),("openDriveUrl",256,128),("opendriveMd5",128,384))
    if native_type is None or not fields or len(fields)!=3 or ctypes.sizeof(native_type)!=512:
        raise ValueError("MAP_IDENTITY_ABI_UNVERIFIED")
    for (name,kind),(key,size,offset) in zip(fields,expected):
        if (name!=key or getattr(kind,"_type_",None) is not ctypes.c_char
                or getattr(kind,"_length_",None)!=size or getattr(native_type,name).offset!=offset):
            raise ValueError("MAP_IDENTITY_ABI_UNVERIFIED")
    value = native_type()
    if service.SoGetHDMapData(value) is not True:
        raise ValueError("MAP_IDENTITY_READ_FAILED")
    identifier,digest,url = (_text(getattr(value,k)) for k in ("openDrive","opendriveMd5","openDriveUrl"))
    if not identifier or not re.fullmatch(r"[0-9a-fA-F]{32}",digest):
        raise ValueError("MAP_IDENTITY_INCOMPLETE")
    return identifier,digest.lower(),url


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise ValueError("MAP_DOCUMENT_REDIRECT_UNSUPPORTED")


class MapDocument(object):
    def __init__(self,clock=None):
        self.clock = clock or time.monotonic
        self.identity,self.data,self.source,self.reason = None,None,"none","MAP_DOCUMENT_UNAVAILABLE"
        self.current = False

    def clear(self,reason):
        self.identity,self.data,self.source,self.reason = None,None,"none",reason
        self.current = False

    @staticmethod
    def selected_identity(service,structs,version):
        try:
            identifier,digest,unused = _native_identity(service,structs,version)
            return identifier,digest
        except Exception:
            return None

    def load(self,service,structs,version,sdk_dir,explicit_file="",allow_download=True,
             loaded_identity=None):
        self.clear("MAP_DOCUMENT_UNAVAILABLE")
        try:
            identifier,digest,url = _native_identity(service,structs,version)
            if loaded_identity is not None and (identifier,digest)!=loaded_identity:
                raise ValueError("SDK_MAP_CHANGED_DURING_LOAD")
            install = os.path.abspath(os.path.join(sdk_dir,"..","..",".."))
            directory = os.path.join(install,"Tools","RoadNetworkRTService")
            deadline = self.clock()+3.
            files = [explicit_file] if explicit_file else glob.iglob(os.path.join(directory,"*.xodr"))
            for index,filename in enumerate(files):
                if index>=MAX_CACHE_FILES:
                    break
                if self.clock()>=deadline:
                    break
                try:
                    if os.path.getsize(filename)>MAX_DOCUMENT_BYTES:
                        continue
                    with open(filename,"rb") as stream:
                        data = stream.read(MAX_DOCUMENT_BYTES+1)
                    if len(data)<=MAX_DOCUMENT_BYTES and hashlib.md5(data).hexdigest()==digest:
                        self._accept(identifier,digest,data,"verified_local_cache")
                        self.verified(service,structs,version)
                        return self.metadata()
                except OSError:
                    continue
            if explicit_file:
                raise ValueError("EXPLICIT_MAP_DOCUMENT_MISSING_OR_DIGEST_MISMATCH")
            if not allow_download or not url:
                raise ValueError("MAP_DOCUMENT_DIGEST_NOT_FOUND")
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme not in ("http","https") or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("MAP_DOCUMENT_URL_UNSUPPORTED")
            request = urllib.request.Request(url,headers={"Accept-Encoding":"identity"})
            deadline = self.clock()+3.
            opener = urllib.request.build_opener(_NoRedirect())
            with opener.open(request,timeout=2.) as stream:
                encoding = stream.headers.get("Content-Encoding","identity")
                if encoding not in ("", "identity"):
                    raise ValueError("MAP_DOCUMENT_ENCODING_UNSUPPORTED")
                size = stream.headers.get("Content-Length")
                if size is not None and (not size.isdigit() or int(size)>MAX_DOCUMENT_BYTES):
                    raise ValueError("MAP_DOCUMENT_TOO_LARGE")
                chunks,total = [],0
                while True:
                    if self.clock()>=deadline:
                        raise ValueError("MAP_DOCUMENT_TIME_BUDGET")
                    part = stream.read(min(65536,MAX_DOCUMENT_BYTES+1-total))
                    if not part:
                        break
                    chunks.append(part)
                    total += len(part)
                    if total>MAX_DOCUMENT_BYTES:
                        raise ValueError("MAP_DOCUMENT_TOO_LARGE")
            data = b"".join(chunks)
            if hashlib.md5(data).hexdigest()!=digest:
                raise ValueError("MAP_DOCUMENT_DIGEST_MISMATCH")
            self._accept(identifier,digest,data,"verified_sdk_url")
            self.verified(service,structs,version)
        except Exception as exc:
            # Exception text may contain signed URLs. Keep only local codes.
            code = str(exc) if isinstance(exc,ValueError) and re.fullmatch(r"[A-Z_]+",str(exc)) else type(exc).__name__
            self.clear(code)
        return self.metadata()

    def _accept(self,identifier,digest,data,source):
        if not data or data.startswith((b"\xff\xfe",b"\xfe\xff")) or b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
            raise ValueError("MAP_DOCUMENT_XML_UNSUPPORTED")
        data.decode("utf-8-sig","strict")
        self.identity,self.data,self.source,self.reason = (identifier,digest),data,source,"SDK_MAP_DOCUMENT_MATCHED"
        self.current = True

    def verified(self,service,structs,version):
        if self.data is None:
            return False
        try:
            identifier,digest,unused = _native_identity(service,structs,version)
            if (identifier,digest)!=self.identity:
                self.clear("SDK_MAP_IDENTITY_CHANGED_RELOAD_REQUIRED")
                return False
            self.current,self.reason = True,"SDK_MAP_DOCUMENT_MATCHED"
            return True
        except Exception:
            # A transient read failure cannot leave a clearance certificate.
            # Keep private immutable bytes for recovery, but withdraw their
            # authority immediately. A later exact identity read can restore it.
            self.current,self.reason = False,"SDK_MAP_IDENTITY_REVALIDATION_FAILED"
            return False

    def metadata(self):
        return dict(verified=self.current and self.data is not None,source=self.source,reason=self.reason,
                    digest=None if self.identity is None else self.identity[1],digest_kind="sdk_md5",
                    size_bytes=0 if self.data is None else len(self.data))
