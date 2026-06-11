"""Dev tool: parse an exported world with nbtlib and sanity check it.
Not bundled into the exe."""

import io
import struct
import sys
import zlib

import nbtlib


def check_level(path):
    f = nbtlib.load(path)  # handles gzip
    data = f["Data"]
    print(f"level.dat ok: '{data['LevelName']}'"
          f" DataVersion={int(data['DataVersion'])}"
          f" spawn=({int(data['SpawnX'])},{int(data['SpawnY'])},"
          f"{int(data['SpawnZ'])})")
    wgs = data["WorldGenSettings"]
    dims = list(wgs["dimensions"].keys())
    print(f"  seed={int(wgs['seed'])} dimensions={dims}")


def check_region(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    assert len(raw) % 4096 == 0, "region not sector aligned"
    n = 0
    statuses = set()
    first = None
    for i in range(1024):
        off = struct.unpack(">I", raw[i * 4:i * 4 + 4])[0]
        sector, count = off >> 8, off & 0xFF
        if sector == 0:
            continue
        n += 1
        pos = sector * 4096
        length, ctype = struct.unpack(">iB", raw[pos:pos + 5])
        assert ctype == 2, f"unexpected compression {ctype}"
        chunk_raw = zlib.decompress(raw[pos + 5:pos + 4 + length])
        tag = nbtlib.File.parse(io.BytesIO(chunk_raw))
        statuses.add(str(tag["Status"]))
        if first is None:
            first = tag
    print(f"{path}: {n} chunks, statuses={statuses}")
    t = first
    secs = t["sections"]
    print(f"  first chunk x={int(t['xPos'])} z={int(t['zPos'])}"
          f" yPos={int(t['yPos'])} sections={len(secs)}"
          f" DataVersion={int(t['DataVersion'])}")
    hm = t["Heightmaps"]["MOTION_BLOCKING"]
    assert len(hm) == 37, f"heightmap longs = {len(hm)}"
    blocks = set()
    biomes = set()
    for s in secs:
        bs = s["block_states"]
        pal = [str(e["Name"]) for e in bs["palette"]]
        blocks.update(pal)
        if len(pal) > 1:
            bits = max(4, (len(pal) - 1).bit_length())
            need = -(-4096 // (64 // bits))
            got = len(bs["data"])
            assert got == need, f"block data longs {got} != {need}"
        else:
            assert "data" not in bs, "palette of 1 must not have data"
        bp = [str(e) for e in s["biomes"]["palette"]]
        biomes.update(bp)
        if len(bp) > 1:
            bits = max(1, (len(bp) - 1).bit_length())
            need = -(-64 // (64 // bits))
            assert len(s["biomes"]["data"]) == need, "biome data size"
        y = int(s["Y"])
        assert -4 <= y <= 19, f"bad section Y {y}"
    print(f"  blocks used: {sorted(blocks)}")
    print(f"  biomes used: {sorted(biomes)}")


if __name__ == "__main__":
    world_dir = sys.argv[1]
    check_level(world_dir + "/level.dat")
    import glob
    for mca in glob.glob(world_dir + "/region/*.mca"):
        check_region(mca)
    print("verify ok")
