from pyboy import PyBoy

pyboy = PyBoy("ROMs/super_breakout.gbc")


# print(pyboy.mb)
while not pyboy.tick():
    if pyboy.frame_count == 10:
        pass
        # print(pyboy.get_memory_value(0x0))
        # print(pyboy.__dir__())
        # print(pyboy.plugin_manager.gamewrapper.__dir__())
        # print(pyboy.mb.ram.internal_ram0)
pyboy.stop()
