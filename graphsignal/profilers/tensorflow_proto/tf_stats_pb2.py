# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: graphsignal/profilers/tensorflow_proto/tf_stats.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n5graphsignal/profilers/tensorflow_proto/tf_stats.proto\x12\x13tensorflow.profiler\"\xa7\x01\n\x0fTfStatsDatabase\x12\x34\n\twith_idle\x18\x04 \x01(\x0b\x32!.tensorflow.profiler.TfStatsTable\x12\x37\n\x0cwithout_idle\x18\x05 \x01(\x0b\x32!.tensorflow.profiler.TfStatsTable\x12\x13\n\x0b\x64\x65vice_type\x18\x06 \x01(\tJ\x04\x08\x01\x10\x02J\x04\x08\x02\x10\x03J\x04\x08\x03\x10\x04\"\x83\x01\n\x0cTfStatsTable\x12;\n\x0ftf_stats_record\x18\x01 \x03(\x0b\x32\".tensorflow.profiler.TfStatsRecord\x12\x19\n\x11host_tf_pprof_key\x18\x02 \x01(\t\x12\x1b\n\x13\x64\x65vice_tf_pprof_key\x18\x03 \x01(\t\"\xbb\x04\n\rTfStatsRecord\x12\x0c\n\x04rank\x18\x01 \x01(\x04\x12\x16\n\x0ehost_or_device\x18\x02 \x01(\t\x12\x0f\n\x07op_type\x18\x03 \x01(\t\x12\x0f\n\x07op_name\x18\x04 \x01(\t\x12\x13\n\x0boccurrences\x18\x05 \x01(\x03\x12\x18\n\x10total_time_in_us\x18\x06 \x01(\x01\x12\x16\n\x0e\x61vg_time_in_us\x18\x07 \x01(\x01\x12\x1d\n\x15total_self_time_in_us\x18\x08 \x01(\x01\x12\x1b\n\x13\x61vg_self_time_in_us\x18\t \x01(\x01\x12*\n\"device_total_self_time_as_fraction\x18\n \x01(\x01\x12\x35\n-device_cumulative_total_self_time_as_fraction\x18\x0b \x01(\x01\x12(\n host_total_self_time_as_fraction\x18\x0c \x01(\x01\x12\x33\n+host_cumulative_total_self_time_as_fraction\x18\r \x01(\x01\x12\x1a\n\x12measured_flop_rate\x18\x0e \x01(\x01\x12\x1a\n\x12measured_memory_bw\x18\x0f \x01(\x01\x12\x1d\n\x15operational_intensity\x18\x10 \x01(\x01\x12\x10\n\x08\x62ound_by\x18\x11 \x01(\t\x12\x10\n\x08is_eager\x18\x12 \x01(\x08\x12\"\n\x1agpu_tensorcore_utilization\x18\x13 \x01(\x01\x62\x06proto3')



_TFSTATSDATABASE = DESCRIPTOR.message_types_by_name['TfStatsDatabase']
_TFSTATSTABLE = DESCRIPTOR.message_types_by_name['TfStatsTable']
_TFSTATSRECORD = DESCRIPTOR.message_types_by_name['TfStatsRecord']
TfStatsDatabase = _reflection.GeneratedProtocolMessageType('TfStatsDatabase', (_message.Message,), {
  'DESCRIPTOR' : _TFSTATSDATABASE,
  '__module__' : 'graphsignal.profilers.tensorflow_proto.tf_stats_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.profiler.TfStatsDatabase)
  })
_sym_db.RegisterMessage(TfStatsDatabase)

TfStatsTable = _reflection.GeneratedProtocolMessageType('TfStatsTable', (_message.Message,), {
  'DESCRIPTOR' : _TFSTATSTABLE,
  '__module__' : 'graphsignal.profilers.tensorflow_proto.tf_stats_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.profiler.TfStatsTable)
  })
_sym_db.RegisterMessage(TfStatsTable)

TfStatsRecord = _reflection.GeneratedProtocolMessageType('TfStatsRecord', (_message.Message,), {
  'DESCRIPTOR' : _TFSTATSRECORD,
  '__module__' : 'graphsignal.profilers.tensorflow_proto.tf_stats_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.profiler.TfStatsRecord)
  })
_sym_db.RegisterMessage(TfStatsRecord)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _TFSTATSDATABASE._serialized_start=79
  _TFSTATSDATABASE._serialized_end=246
  _TFSTATSTABLE._serialized_start=249
  _TFSTATSTABLE._serialized_end=380
  _TFSTATSRECORD._serialized_start=383
  _TFSTATSRECORD._serialized_end=954
# @@protoc_insertion_point(module_scope)