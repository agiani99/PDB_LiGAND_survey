"""Tiny localhost HTTP server that draws ligand structures on the fly with RDKit.

GET /mol/<COMP_ID>.svg  ->  2D depiction generated from the component SMILES.
Used by the Bokeh hover tooltip so no images have to be precomputed or embedded.
Bound to 127.0.0.1 only (the app and the browser must be on the same machine).
"""
from __future__ import annotations

import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

SMILES: dict[str, str] = {}
_EMPTY = b'<svg xmlns="http://www.w3.org/2000/svg" width="200" height="150"/>'


@lru_cache(maxsize=20000)
def draw_svg(comp_id: str, w: int = 220, h: int = 160) -> bytes:
    smi = SMILES.get(comp_id)
    if not smi:
        return _EMPTY
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return _EMPTY
    rdDepictor.Compute2DCoords(mol)
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    d.drawOptions().clearBackground = False
    d.DrawMolecule(mol)
    d.FinishDrawing()
    return d.GetDrawingText().encode()


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        name = self.path.rsplit("/", 1)[-1].split("?")[0]
        comp = name[:-4] if name.endswith(".svg") else name
        body = draw_svg(comp)
        self.send_response(200)
        self.send_header("Content-Type", "image/svg+xml")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence
        pass


_server = None
_lock = threading.Lock()


def start() -> int:
    """Start (once per process) and return the port."""
    global _server
    with _lock:
        if _server is None:
            _server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
            threading.Thread(target=_server.serve_forever, daemon=True).start()
        return _server.server_address[1]
