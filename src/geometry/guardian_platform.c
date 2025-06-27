/// guardian_platform.c
/// Cross-platform system interface implementations.

#include "guardian_platform.h"

#ifdef _WIN32
#include <windows.h>
#include <conio.h>
#else
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
#endif

// Mutex Functions

/// Initialize a mutex.
mutex_t* guardian_mutex_init() {
#ifdef _WIN32
    return CreateMutex(NULL, FALSE, NULL);
#else
    mutex_t* mtx = (mutex_t*)malloc(sizeof(mutex_t));
    if (mtx) pthread_mutex_init(mtx, NULL);
    return mtx;
#endif
}

/// Destroy a mutex.
void guardian_mutex_destroy(mutex_t* mutex) {
#ifdef _WIN32
    if (mutex) CloseHandle(mutex);
#else
    if (mutex) {
        pthread_mutex_destroy(mutex);
        free(mutex);
    }
#endif
}

// Console Color and Cursor Functions

void guardian_console_set_color(int color_code) {
#ifdef _WIN32
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, color_code);
#else
    printf("\033[%dm", color_code);
#endif
}

void guardian_console_reset_color() {
#ifdef _WIN32
    guardian_console_set_color(7);
#else
    printf("\033[0m");
#endif
}

void guardian_console_move_cursor(int row, int col) {
#ifdef _WIN32
    COORD coord = {(SHORT)col, (SHORT)row};
    SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), coord);
#else
    printf("\033[%d;%dH", row + 1, col + 1);
#endif
}

void guardian_console_clear() {
#ifdef _WIN32
    system("cls");
#else
    printf("\033[2J\033[H");
#endif
}

void guardian_console_hide_cursor() {
#ifdef _WIN32
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_CURSOR_INFO cursorInfo;
    GetConsoleCursorInfo(hConsole, &cursorInfo);
    cursorInfo.bVisible = FALSE;
    SetConsoleCursorInfo(hConsole, &cursorInfo);
#else
    printf("\033[?25l");
#endif
}

void guardian_console_show_cursor() {
#ifdef _WIN32
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_CURSOR_INFO cursorInfo;
    GetConsoleCursorInfo(hConsole, &cursorInfo);
    cursorInfo.bVisible = TRUE;
    SetConsoleCursorInfo(hConsole, &cursorInfo);
#else
    printf("\033[?25h");
#endif
}

// Raw Keyboard Input and Device I/O

int guardian_key_pressed() {
#ifdef _WIN32
    return _kbhit();
#else
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
#endif
}

int guardian_get_char() {
#ifdef _WIN32
    return _getch();
#else
    struct termios oldt, newt;
    int ch;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ch;
#endif
}
