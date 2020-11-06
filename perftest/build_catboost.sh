#!/bin/sh

TOP=$(cd $(dirname "$0"); pwd)

BUILD="$TOP/build"
CATBOOST="$BUILD/catboost"

die() {
    echo "Error: $*" 1>&2
    exit 1
}

if [ -f "$BUILD/libcatboostmodel.so" ]; then
    echo "Library is already built. If you want to rebuild it remove libcatboostmodel.so"
    exit 0
fi

mkdir -p "$BUILD"
if [ \! -d "$CATBOOST" ]; then
    (cd "$BUILD"; git clone https://github.com/catboost/catboost.git) || die "git clone failed"
fi


(cd "$CATBOOST"; ./ya make -r catboost/libs/model_interface) || die "build failed"
cp $(readlink "$CATBOOST/catboost/libs/model_interface/libcatboostmodel.so") "$BUILD/libcatboostmodel.so"
cp "$CATBOOST/catboost/libs/model_interface/c_api.h" "$BUILD/catboost_capi.h"

echo "Catboost library has been successfully built!"
