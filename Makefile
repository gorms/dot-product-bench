CXX      = g++
CXXFLAGS = -O2 -Wall -Wextra -mavx2 -mfma -std=c++23 $(shell pkg-config --cflags eigen3)

bench: main.o dot_avx512.o
	$(CXX) $(CXXFLAGS) -o bench main.o dot_avx512.o

main.o: main.cpp dot_avx512.hpp
	$(CXX) $(CXXFLAGS) -c -o main.o main.cpp

dot_avx512.o: dot_avx512.cpp dot_avx512.hpp
	$(CXX) $(CXXFLAGS) -mavx512f -c -o dot_avx512.o dot_avx512.cpp

clean:
	rm -f bench main.o dot_avx512.o

.PHONY: clean
