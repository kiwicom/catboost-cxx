CatBoost applier library
-------------------------
![Test status:](https://github.com/kiwicom/catboost-cxx/workflows/CMake/badge.svg)

This library allows to apply Yandex Catboost models without adding huge applier library dependency.
Models should be saved in JSON format.

The main purpose of the library is to have possibility to apply models with reasonable performance and so
we are using SSE4.1 for acceleration of the code. For platforms without SSE instructions library has plain
C++ implementation.

This library is being used in Kiwi.com for runtime predictions.

Usage
=====
```cpp
#include <catboost.hpp>

void predict(const std::vector<float>& x) {
    catboost::Model model{"model.json"};

    double y = model.apply(x);

    std::cout << "Predicted value is " << y << std::endl;
}
```

Project could be built using CMake.

Performance
===========
As could be seen from perf.txt this library is faster than Yandex implementation on single predictions but ~3 times slower on buckets. I'll try to be even faster, but not now.

How it works
============
My approach is very simple: I group trees by 4 and then apply model in parallel using SSE instructions. For single trees it is done using multilevel apply if depth is greater than 3. To do this I group trees by depth and write them into specially aligned binary array.

I've tried to use AVX2 instructions and found out that performance is the same. So I decided to not merge these changes because becomes more complex without any improvement.

Limitations
===========
Categorical features and multitarget models are not supported now. In the future releases I'm planning to
add this functional.

Currently library only supports SMD instructions on platforms other than x86-64. ARM v6+ support is planned using sse2neon header.

Windows version is not tested yet, but hopefully should work.

Binary models from CatBoost library could not be loaded. This problem is more complex than it looks at first glance. Binary form of the model is prepared for fast applier and I'd need to convert it. Moreover I don't want to have protobuf dependency.

Testing
=======
Library contains unit tests and performance test.
``` bash
make test # build and run tests
make perf # build and run performance tests
```

Performance results could be found in perftest/perf.txt

If you want to add new testing dataset for performance testing you can use `perftest/train_models/train_models.py` script.
At this time we are using following datasets for testing:
 1. MSRank dataset
 2. Credit Germany dataset
 3. Normalized form of codrna

References
==========
```
@article{DBLP:journals/corr/QinL13,
  author    = {Tao Qin and
               Tie{-}Yan Liu},
  title     = {Introducing {LETOR} 4.0 Datasets},
  journal   = {CoRR},
  volume    = {abs/1306.2597},
  year      = {2013},
  url       = {http://arxiv.org/abs/1306.2597},
  timestamp = {Mon, 01 Jul 2013 20:31:25 +0200},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/QinL13},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```

```
German Credit data
Author: Dr. Hans Hofmann
Source: UCI - 1994
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
```

```
Normalized form of codrna (351)

Author: Andrew V Uzilov","Joshua M Keegan","David H Mathews.
Source: original -
Please cite: [AVU06a] Andrew V Uzilov, Joshua M Keegan, and David H Mathews. Detection of non-coding RNAs on the basis of predicted secondary structure formation free energy change. BMC Bioinformatics, 7(173), 2006.
```
