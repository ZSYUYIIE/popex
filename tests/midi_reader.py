"""Read the Standard MIDI Files that PopEx writes, for tests."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field


@dataclass
class MidiTrack:
    name: str | None = None
    # (tick, status, data bytes) for channel messages, in file order
    channel: list[tuple[int, int, bytes]] = field(default_factory=list)
    # (tick, meta type, data) in file order
    meta: list[tuple[int, int, bytes]] = field(default_factory=list)
    end_tick: int = 0


@dataclass
class MidiFile:
    format: int
    division: int
    tracks: list[MidiTrack]


def _varlen(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if byte < 0x80:
            return value, offset


def read_midi(payload: bytes) -> MidiFile:
    assert payload[:4] == b"MThd"
    length, file_format, count, division = struct.unpack_from(">IHHH", payload, 4)
    assert length == 6
    offset = 14
    tracks: list[MidiTrack] = []
    for _ in range(count):
        assert payload[offset : offset + 4] == b"MTrk"
        size = struct.unpack_from(">I", payload, offset + 4)[0]
        data = payload[offset + 8 : offset + 8 + size]
        offset += 8 + size
        track = MidiTrack()
        cursor = tick = 0
        while cursor < len(data):
            delta, cursor = _varlen(data, cursor)
            tick += delta
            status = data[cursor]
            cursor += 1
            if status == 0xFF:
                meta_type = data[cursor]
                meta_size, cursor = _varlen(data, cursor + 1)
                body = data[cursor : cursor + meta_size]
                cursor += meta_size
                track.meta.append((tick, meta_type, body))
                if meta_type == 0x03:
                    track.name = body.decode("ascii")
                if meta_type == 0x2F:
                    track.end_tick = tick
                    break
                continue
            assert status & 0x80, "running status is not used by PopEx"
            width = 1 if status & 0xF0 in (0xC0, 0xD0) else 2
            track.channel.append((tick, status, data[cursor : cursor + width]))
            cursor += width
        assert cursor == len(data), "data after end of track"
        tracks.append(track)
    assert offset == len(payload)
    return MidiFile(file_format, division, tracks)


def note_events(payload: bytes) -> list[tuple[int, int, int, int]]:
    """``(tick, status, key, velocity)`` note-ons and offs of every track, merged by tick."""
    merged = []
    for index, track in enumerate(read_midi(payload).tracks):
        for position, (tick, status, data) in enumerate(track.channel):
            if status & 0xF0 in (0x80, 0x90):
                merged.append((tick, index, position, status, data[0], data[1]))
    merged.sort()
    return [(tick, status, key, velocity) for tick, _i, _p, status, key, velocity in merged]
