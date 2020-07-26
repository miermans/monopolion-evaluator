# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: game_outcome.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='game_outcome.proto',
  package='monopolion.label.protobuf',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x12game_outcome.proto\x12\x19monopolion.label.protobuf\"X\n\x06Player\x12\x10\n\x08position\x18\x01 \x01(\x05\x12\x0c\n\x04\x63\x61sh\x18\x02 \x01(\x11\x12\x10\n\x08isInJail\x18\x03 \x01(\x08\x12\x1c\n\x14remainingTurnsInJail\x18\x04 \x01(\x05\"F\n\rPropertyState\x12\x0f\n\x07isOwned\x18\x01 \x01(\x08\x12\r\n\x05owner\x18\x02 \x01(\x11\x12\x15\n\rbuildingCount\x18\x03 \x01(\x05\"\xb3\x03\n\tGameState\x12\x39\n\x05state\x18\x01 \x01(\x0e\x32*.monopolion.label.protobuf.GameState.State\x12\x0c\n\x04\x64ie1\x18\x02 \x01(\x05\x12\x0c\n\x04\x64ie2\x18\x03 \x01(\x05\x12\x32\n\x07players\x18\x04 \x03(\x0b\x32!.monopolion.label.protobuf.Player\x12@\n\x0epropertyStates\x18\x05 \x03(\x0b\x32(.monopolion.label.protobuf.PropertyState\"\xd8\x01\n\x05State\x12\x08\n\x04ROLL\x10\x00\x12\x08\n\x04MOVE\x10\x01\x12\x11\n\rLAND_ON_SPACE\x10\x02\x12\x16\n\x12\x42UY_SELL_POST_ROLL\x10\x03\x12\x0f\n\x0b\x45ND_OF_ROLL\x10\x04\x12\x0f\n\x0bNEXT_PLAYER\x10\x05\x12\x15\n\x11\x42UY_SELL_PRE_ROLL\x10\x06\x12\x0c\n\x08PRE_ROLL\x10\x07\x12\x0b\n\x07IN_JAIL\x10\x08\x12\x11\n\rJAIL_DECISION\x10\t\x12\r\n\tJAIL_ROLL\x10\n\x12\x0c\n\x08JAIL_FEE\x10\x0b\x12\x0c\n\x08GAME_END\x10\x63\"]\n\x0bGameOutcome\x12\x15\n\rwinningPlayer\x18\x01 \x01(\x05\x12\x37\n\tgameState\x18\x02 \x01(\x0b\x32$.monopolion.label.protobuf.GameStateb\x06proto3'
)



_GAMESTATE_STATE = _descriptor.EnumDescriptor(
  name='State',
  full_name='monopolion.label.protobuf.GameState.State',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='ROLL', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='MOVE', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='LAND_ON_SPACE', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BUY_SELL_POST_ROLL', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='END_OF_ROLL', index=4, number=4,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='NEXT_PLAYER', index=5, number=5,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BUY_SELL_PRE_ROLL', index=6, number=6,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='PRE_ROLL', index=7, number=7,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='IN_JAIL', index=8, number=8,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='JAIL_DECISION', index=9, number=9,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='JAIL_ROLL', index=10, number=10,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='JAIL_FEE', index=11, number=11,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='GAME_END', index=12, number=99,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=431,
  serialized_end=647,
)
_sym_db.RegisterEnumDescriptor(_GAMESTATE_STATE)


_PLAYER = _descriptor.Descriptor(
  name='Player',
  full_name='monopolion.label.protobuf.Player',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='position', full_name='monopolion.label.protobuf.Player.position', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cash', full_name='monopolion.label.protobuf.Player.cash', index=1,
      number=2, type=17, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='isInJail', full_name='monopolion.label.protobuf.Player.isInJail', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='remainingTurnsInJail', full_name='monopolion.label.protobuf.Player.remainingTurnsInJail', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=49,
  serialized_end=137,
)


_PROPERTYSTATE = _descriptor.Descriptor(
  name='PropertyState',
  full_name='monopolion.label.protobuf.PropertyState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='isOwned', full_name='monopolion.label.protobuf.PropertyState.isOwned', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='owner', full_name='monopolion.label.protobuf.PropertyState.owner', index=1,
      number=2, type=17, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='buildingCount', full_name='monopolion.label.protobuf.PropertyState.buildingCount', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=139,
  serialized_end=209,
)


_GAMESTATE = _descriptor.Descriptor(
  name='GameState',
  full_name='monopolion.label.protobuf.GameState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='state', full_name='monopolion.label.protobuf.GameState.state', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='die1', full_name='monopolion.label.protobuf.GameState.die1', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='die2', full_name='monopolion.label.protobuf.GameState.die2', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='players', full_name='monopolion.label.protobuf.GameState.players', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='propertyStates', full_name='monopolion.label.protobuf.GameState.propertyStates', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _GAMESTATE_STATE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=212,
  serialized_end=647,
)


_GAMEOUTCOME = _descriptor.Descriptor(
  name='GameOutcome',
  full_name='monopolion.label.protobuf.GameOutcome',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='winningPlayer', full_name='monopolion.label.protobuf.GameOutcome.winningPlayer', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='gameState', full_name='monopolion.label.protobuf.GameOutcome.gameState', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=649,
  serialized_end=742,
)

_GAMESTATE.fields_by_name['state'].enum_type = _GAMESTATE_STATE
_GAMESTATE.fields_by_name['players'].message_type = _PLAYER
_GAMESTATE.fields_by_name['propertyStates'].message_type = _PROPERTYSTATE
_GAMESTATE_STATE.containing_type = _GAMESTATE
_GAMEOUTCOME.fields_by_name['gameState'].message_type = _GAMESTATE
DESCRIPTOR.message_types_by_name['Player'] = _PLAYER
DESCRIPTOR.message_types_by_name['PropertyState'] = _PROPERTYSTATE
DESCRIPTOR.message_types_by_name['GameState'] = _GAMESTATE
DESCRIPTOR.message_types_by_name['GameOutcome'] = _GAMEOUTCOME
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Player = _reflection.GeneratedProtocolMessageType('Player', (_message.Message,), {
  'DESCRIPTOR' : _PLAYER,
  '__module__' : 'game_outcome_pb2'
  # @@protoc_insertion_point(class_scope:monopolion.label.protobuf.Player)
  })
_sym_db.RegisterMessage(Player)

PropertyState = _reflection.GeneratedProtocolMessageType('PropertyState', (_message.Message,), {
  'DESCRIPTOR' : _PROPERTYSTATE,
  '__module__' : 'game_outcome_pb2'
  # @@protoc_insertion_point(class_scope:monopolion.label.protobuf.PropertyState)
  })
_sym_db.RegisterMessage(PropertyState)

GameState = _reflection.GeneratedProtocolMessageType('GameState', (_message.Message,), {
  'DESCRIPTOR' : _GAMESTATE,
  '__module__' : 'game_outcome_pb2'
  # @@protoc_insertion_point(class_scope:monopolion.label.protobuf.GameState)
  })
_sym_db.RegisterMessage(GameState)

GameOutcome = _reflection.GeneratedProtocolMessageType('GameOutcome', (_message.Message,), {
  'DESCRIPTOR' : _GAMEOUTCOME,
  '__module__' : 'game_outcome_pb2'
  # @@protoc_insertion_point(class_scope:monopolion.label.protobuf.GameOutcome)
  })
_sym_db.RegisterMessage(GameOutcome)


# @@protoc_insertion_point(module_scope)