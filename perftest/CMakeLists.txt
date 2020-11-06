
IF(ENABLE_PERF)
    ADD_EXECUTABLE(perf_test
        main.cpp
        build/catboost_capi.h
    )
    TARGET_LINK_LIBRARIES(perf_test catboost ${CMAKE_DL_LIBS})

    ADD_CUSTOM_COMMAND(
        OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/build/catboost_capi.h"
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/build_catboost.sh"
    )
ENDIF()
