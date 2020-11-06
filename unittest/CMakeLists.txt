
ADD_EXECUTABLE(unittest catboost_test.cpp)

TARGET_LINK_LIBRARIES(unittest catboost)

ADD_TEST(
    NAME catboosttest
    COMMAND unittest -v -d ${CMAKE_CURRENT_SOURCE_DIR}
)

