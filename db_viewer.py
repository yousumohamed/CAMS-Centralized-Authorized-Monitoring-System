"""
CAMS – Encrypted DB Viewer (CLI)
Decrypts and prints detection/system logs in a readable table format.

Usage
-----
  python db_viewer.py detections [--limit 50]
  python db_viewer.py system     [--limit 100]
  python db_viewer.py stats
"""
import argparse
import sys
from db import fetch_detections, fetch_system_logs, init_db


def _hr(char="─", width=100):
    print(char * width)


def cmd_detections(limit: int):
    rows = fetch_detections(limit)
    if not rows:
        print("No detections logged yet.")
        return

    _hr("═")
    print(f"{'ID':>5}  {'Timestamp':19}  {'Camera':10}  {'Name':14}  "
          f"{'Conf%':6}  {'Lat':10}  {'Lon':11}  {'Crop File'}")
    _hr()
    for r in rows:
        crop = r["crop_path"].split("\\")[-1].split("/")[-1]
        print(f"{r['id']:>5}  {r['timestamp']:19}  {r['camera_id']:10}  "
              f"{r['camera_name']:14}  {r['confidence']:>6.2f}  "
              f"{float(r['gps_lat']):>10.5f}  {float(r['gps_lon']):>11.5f}  {crop}")
    _hr("═")
    print(f"  {len(rows)} record(s) shown  (GPS coordinates decrypted in memory)")
    print()


def cmd_system(limit: int):
    rows = fetch_system_logs(limit)
    if not rows:
        print("No system log entries yet.")
        return

    _hr("═")
    print(f"{'ID':>5}  {'Timestamp':19}  {'Level':8}  {'Message'}")
    _hr()
    for r in rows:
        print(f"{r['id']:>5}  {r['timestamp']:19}  {r['level']:8}  {r['message']}")
    _hr("═")
    print(f"  {len(rows)} entry(ies) shown\n")


def cmd_stats():
    detections = fetch_detections(10_000)
    by_camera : dict[str, int] = {}
    for d in detections:
        by_camera[d["camera_id"]] = by_camera.get(d["camera_id"], 0) + 1

    print("\n📊  Detection Statistics")
    _hr()
    total = len(detections)
    print(f"  Total detections logged : {total}")
    if total:
        top = sorted(by_camera.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top camera              : {top[0][0]} ({top[0][1]} detections)")
        avg_conf = sum(d["confidence"] for d in detections) / total
        print(f"  Average confidence      : {avg_conf:.1f} %")
    print()
    print(f"{'Camera':<12} {'Detections':>12}")
    _hr("-", 26)
    for cam, cnt in sorted(by_camera.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * min(cnt, 40)
        print(f"  {cam:<10} {cnt:>8}  {bar}")
    print()


def main():
    init_db()
    p = argparse.ArgumentParser(
        description="CAMS encrypted database viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd")

    det = sub.add_parser("detections", help="List target detection events")
    det.add_argument("--limit", type=int, default=50)

    sys_p = sub.add_parser("system", help="List system log entries")
    sys_p.add_argument("--limit", type=int, default=100)

    sub.add_parser("stats", help="Show aggregate statistics")

    args = p.parse_args()
    if args.cmd == "detections":
        cmd_detections(args.limit)
    elif args.cmd == "system":
        cmd_system(args.limit)
    elif args.cmd == "stats":
        cmd_stats()
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
