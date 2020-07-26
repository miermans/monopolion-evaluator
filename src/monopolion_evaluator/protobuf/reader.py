import gzip
from monopolion_evaluator.protobuf import game_outcome_pb2
from google.protobuf.internal.decoder import _DecodeVarint32


def parse_delimited_file(filename: str, decompress: bool = True) -> game_outcome_pb2.GameOutcome:
    game_outcomes = []
    open_function = gzip.open if decompress else open

    with open_function(filename, "rb") as f:
        buf = f.read()
        n = 0
        while n < len(buf):
            msg_len, new_pos = _DecodeVarint32(buf, n)
            n = new_pos
            msg_buf = buf[n:n + msg_len]
            n += msg_len
            game_outcome = game_outcome_pb2.GameOutcome()
            game_outcome.ParseFromString(msg_buf)
            game_outcomes.append(game_outcome)

    return game_outcomes
