all: devel

MKARGS ?= -j 2

devel: build_debug
	(cd build_debug; make $(MKARGS))

release: build_release
	(cd build_release; make $(MKARGS))

devel_nosse: build_debug_nosse
	(cd build_debug_nosse; make $(MKARGS))

release_nosse: build_release_nosse
	(cd build_release_nosse; make $(MKARGS))

unittest: build_unittest
	(cd build_unittest; make $(MKARGS))

unittest_nosse: build_unittest_nosse
	(cd build_unittest_nosse; make $(MKARGS))

build_debug: CMakeLists.txt
	rm -rf build_debug
	mkdir build_debug
	(cd build_debug; cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_PERF=ON ../)
	[ -f compile_commands.json ] || ln -s build_debug/compile_commands.json .

build_unittest: CMakeLists.txt
	rm -rf build_unittest
	mkdir build_unittest
	(cd build_unittest; cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_PERF=OFF ../)
	[ -f compile_commands.json ] || ln -s build_unittest/compile_commands.json .

build_unittest_nosse: CMakeLists.txt
	rm -rf build_unittest_nosse
	mkdir build_unittest_nosse
	(cd build_unittest_nosse; cmake -DDISABLE_SSE=ON -DCMAKE_BUILD_TYPE=Debug -DENABLE_PERF=OFF ../)
	[ -f compile_commands.json ] || ln -s build_unittest_nosse/compile_commands.json .

build_debug_nosse: CMakeLists.txt
	rm -rf build_debug_nosse
	mkdir build_debug_nosse
	(cd build_debug_nosse; cmake -DDISABLE_SSE=ON -DCMAKE_BUILD_TYPE=Debug -DENABLE_PERF=ON ../)
	[ -f compile_commands.json ] || ln -s build_debug_nosse/compile_commands.json .

build_release: CMakeLists.txt
	rm -rf build_release
	mkdir build_release
	(cd build_release; cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PERF=ON ../)
	[ -f compile_commands.json ] || ln -s build_release/compile_commands.json .

build_release_nosse: CMakeLists.txt
	rm -rf build_release_nosse
	mkdir build_release_nosse
	(cd build_release_nosse; cmake -DDISABLE_SSE=ON -DCMAKE_BUILD_TYPE=Release -DENABLE_PERF=ON ../)
	[ -f compile_commands.json ] || ln -s build_release_nosse/compile_commands.json .

test:
	make devel
	./build_debug/unittest/unittest -d ./unittest

test_nosse:
	make devel_nosse
	./build_debug_nosse/unittest/unittest -d ./unittest

perf:
	make release
	(cd ./perftest; ../build_release/perftest/perf_test)

perf_nosse:
	make release_nosse
	(cd ./perftest; ../build_release_nosse/perftest/perf_test)

perftxt:
	make release
	make release_nosse
	echo "WITH SSE:" > ./perftest/perf.txt
	(cd ./perftest; ../build_release/perftest/perf_test) >> ./perftest/perf.txt 2>&1
	echo "WITHOUT SSE:" >> ./perftest/perf.txt
	(cd ./perftest; ../build_release_nosse/perftest/perf_test) >> ./perftest/perf.txt 2>&1
	git add ./perftest/perf.txt
	echo "You can now commit perf.txt"

indent:
	# For indentation I'm using clang-format
	for file in include/catboost.hpp src/catboost.cpp src/vec4.hpp; do clang-format -style=file -i $$file; done

clean:
	rm -rf build_debug
	rm -rf build_release
	rm -f compile_commands.json

.PHONY:
	devel release all clean
