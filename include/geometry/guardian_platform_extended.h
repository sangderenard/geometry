/// extended_platform_delocalization.h
/// Cross-platform device access (input, joystick, etc.)

#ifndef EXTENDED_PLATFORM_DELOCALIZATION_H
#define EXTENDED_PLATFORM_DELOCALIZATION_H

#ifdef _WIN32
#include <windows.h>
#else
#include <linux/input.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ========================================
// 📟 Device Enumeration
// ========================================
typedef enum {
    GUARDIAN_DEVICE_KEYBOARD,
    GUARDIAN_DEVICE_MOUSE,
    GUARDIAN_DEVICE_JOYSTICK,
    GUARDIAN_DEVICE_UNKNOWN
} guardian_device_type_t;

typedef struct {
    char name[256];
    guardian_device_type_t type;
    int id;
    int fd;
} guardian_input_device_t;

int guardian_list_input_devices(guardian_input_device_t* devices, int max_devices);
guardian_device_type_t guardian_identify_device_type(const char* path);

// ========================================
// 🎮 Input Access
// ========================================
int guardian_read_input_event(const guardian_input_device_t* device);
int guardian_open_device(guardian_input_device_t* device);
void guardian_close_device(guardian_input_device_t* device);

#ifdef __cplusplus
}
#endif

#endif // EXTENDED_PLATFORM_DELOCALIZATION_H
