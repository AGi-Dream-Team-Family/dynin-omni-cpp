# examples/dynin-voice - standalone text-to-speech and units-to-speech
# tools for Dynin-Omni checkpoints. See README.md in this directory for what
# each target does.

# dynin_units.h/.cpp (the S2U speech-to-unit encoder) is not called by
# either tool below -- dynin-u2s takes units in, dynin-t2s-kvblock produces
# units directly from the diffusion decode -- but it is compiled into
# dynin-u2s here so it is built and link-checked as part of this example set.
set(TARGET dynin-u2s)
add_executable(${TARGET} dynin-u2s.cpp dynin_vocoder.h dynin_vocoder.cpp dynin_units.h dynin_units.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET dynin-t2s-kvblock)
add_executable(${TARGET} dynin-t2s-kvblock.cpp dynin_vocoder.h dynin_vocoder.cpp dynin_vocab.h dynin_ears.h)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
