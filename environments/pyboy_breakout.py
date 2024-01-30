# https://github.com/Baekalfen/PyBoy/

from pyboy import PyBoy
import io
state_object = io.BytesIO()
state_object.seek(0)

# from pyboy.api.memory_scanner import DynamicComparisonType

pyboy = PyBoy("ROMs/super_breakout.gbc")


while not pyboy.tick():
    if pyboy.frame_count == 10:
        # value = pyboy.get_memory_value(0x0)
        pyboy.save_state(state_object)
        break
        # print(pyboy.__dir__())
        # print(pyboy.plugin_manager.gamewrapper.__dir__())
        # print(pyboy.mb.ram.internal_ram0)
pyboy.stop()