# EM probe setup related configurations

# ========== EM SETTINGS ================================
# Time that you wait to restart the injection after a reboot pin reboot
EM_AFTER_REBOOT_SLEEPING_TIME = 1
# Max sequential error defined for EM
EM_MAX_SEQUENTIALLY_ERRORS = 2
# Range for Delays -- it is code related
# It's the delay of the EM pulse
EM_DELAY_RANGE = range(100, 2000, 100)
# Amplitude must be a range
EM_AMPLITUDE_RANGE = range(-60, -70, -1)

# EM AVRK4
EM_AVRK4_CONNECTION_IP = "192.168.0.101"
EM_AVRK4_CONNECTION_PORT = 23
EM_AVRK4_CONNECTION_CONF = None
# ======================================================
