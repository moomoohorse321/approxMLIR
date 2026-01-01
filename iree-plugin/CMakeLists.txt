# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Approx Input Plugin for IREE
# This plugin registers the approx dialect and passes with IREE
iree_add_all_subdirs()

iree_compiler_register_plugin(
  PLUGIN_ID
    input_approx
  TARGET
    ::registration
)

iree_cc_library(
  NAME
    registration
  SRCS
    "PluginRegistration.cpp"
  DEPS
    MLIRapprox
    MLIRApproxTransforms
    MLIRIR
    MLIRPass
    MLIRSupport
    iree::compiler::PluginAPI
  PUBLIC
)