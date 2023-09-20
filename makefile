test:
	gcc -Wall ./tests/*.c ./src/*.c -lm -lcunit -o ./build/testRunner.out -I./include -D TEST_BUILD -pthread
	./build/testRunner.out
run:
	gcc -Wall ./src/*.c -o ./build/main.out -I./include -pthread -lm
	./build/main.out