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

build_debug: CMakeLists.txt
	rm -rf build_debug
	mkdir build_debug
	(cd build_debug; cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_PERF=ON ../)
	[ -f compile_commands.json ] || ln -s build_debug/compile_commands.json .

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

clean:
	rm -rf build_debug
	rm -rf build_release
	rm -f compile_commands.json

.PHONY:
	devel release all clean
