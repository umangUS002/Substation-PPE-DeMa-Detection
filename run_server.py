"""
Industrial Substation Multi-Camera PPE Detection Server Launcher.
Starts the FastAPI Server, WebSockets, and 24 Industrial Camera Streams.
"""

import sys
import os
import uvicorn

if __name__ == "__main__":
    print("=" * 75)
    print("[SYSTEM] INDUSTRIAL SUBSTATION SAFETY & MULTI-CAMERA PPE MONITORING SYSTEM")
    print("=" * 75)
    print(" Active Channels : 24 Industrial Plant Cameras")
    print(" Target PPE      : Helmet, Gloves, Footwear (+ Vest & Mask)")
    print(" Engine Protocol  : 3-Step DeMa (Determination, Matching, Warning)")
    print(" Dashboard URL   : http://localhost:8000")
    print(" REST API Docs   : http://localhost:8000/docs")
    print("=" * 75)

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
