# Install script for directory: /data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/output")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_api/lib/libcust_opapi.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_api/lib/libcust_opapi.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_api/lib/libcust_opapi.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_api/lib" TYPE SHARED_LIBRARY FILES "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/build/libcust_opapi.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_api/lib/libcust_opapi.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_api/lib/libcust_opapi.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_api/lib/libcust_opapi.so"
         OLD_RPATH "/usr/local/Ascend/ascend-toolkit/latest/lib64:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_api/lib/libcust_opapi.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_proto/lib/linux/aarch64" TYPE SHARED_LIBRARY FILES "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/build/libcust_opsproto_rt2.0.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so"
         OLD_RPATH "/usr/local/Ascend/ascend-toolkit/latest/lib64:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64" TYPE SHARED_LIBRARY FILES "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/build/libcust_opmaster_rt2.0.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"
         OLD_RPATH "/usr/local/Ascend/ascend-toolkit/latest/lib64:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_impl/ai_core/tbe/op_tiling" TYPE FILE FILES "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/build/compat/liboptiling.so")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/build/src/utils/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/build/tests/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/build/examples/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/build/src/transformer/flash_attention_score/ophost/cmake_install.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_proto/inc" TYPE FILE OPTIONAL FILES "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/build/autogen/inner/flash_attention_score_proto.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_impl/ai_core/tbe/foly_attn_impl/dynamic" TYPE FILE OPTIONAL FILES "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/build/impl/dynamic/flash_attention_score.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_impl/ai_core/tbe/foly_attn_impl/ascendc/common" TYPE DIRECTORY FILES "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/src/utils/inc/kernel/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_impl/ai_core/tbe/foly_attn_impl/ascendc/flash_attention_score" TYPE FILE OPTIONAL FILES
    "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/src/transformer/flash_attention_score/flash_attention_score.cpp"
    "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/src/transformer/flash_attention_score/flash_attention_score_bn2gs1s2_b.h"
    "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/src/transformer/flash_attention_score/flash_attention_score_common.h"
    "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/src/transformer/flash_attention_score/flash_attention_score_drop_mask_adapter.h"
    "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/src/transformer/flash_attention_score/flash_attention_score_empty_tensor.h"
    "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/src/transformer/flash_attention_score/flash_attention_score_s1_bn2gs1.h"
    "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/src/transformer/flash_attention_score/flash_attention_score_s1s2_bn2gs1.h"
    "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/src/transformer/flash_attention_score/flash_attention_var_len_score.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn/op_impl/ai_core/tbe/config/ascend910_93" TYPE FILE OPTIONAL FILES "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/build/autogen/aic-ascend910_93-ops-info.json")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE DIRECTORY PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ FILES "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/build/scripts/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/foly_attn" TYPE FILE FILES "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/build/version.info")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/data/Fused_Floyd_Attn_AscendC/cq/foly_attn_proj/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
