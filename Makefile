all: devel

MKARGS ?= -j 2

devel: build_debug
	(cd build_debug; make $(MKARGS))

release: build_release
	(cd build_release; make $(MKARGS))

build_debug: CMakeLists.txt
	rm -rf build_debug
	mkdir build_debug
	(cd build_debug; cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_PERF=ON ../)
	[ -f compile_commands.json ] || ln -s build_debug/compile_commands.json .

build_release: CMakeLists.txt
	rm -rf build_release
	mkdir build_release
	(cd build_release; cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PERF=ON ../)
	[ -f compile_commands.json ] || ln -s build_release/compile_commands.json .

test:
	make devel
	./build_debug/unittest/unittest -d ./unittest

perf:
	make release
	(cd ./perftest; ../build_release/perftest/perf_test)

clean:
	rm -rf build_debug
	rm -rf build_release
	rm -f compile_commands.json

.PHONY:
	devel release all clean
