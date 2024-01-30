## Getting Started

Clone the repo (with submodules):
`git clone --recurse-submodules -j8 https://github.com/persuck/memory-pixel-interp.git`

If you already cloned the repo but need to clone the submodules:
`git submodule update --init --recursive`

Install deps:
`pip install -r requirements.txt`

Or make a virtual env:
`python -m venv env && source ./env/bin/activate && pip install -r requirements.txt`

## Summary

Core components of this project:

- __memory agent__: Uses a simplified view of the internal memory (raw 
RAM) of the game "breakout" as state. 

- __pixel agent__: Uses downscaled, greyscale screen pixels of the game "breakout" as state.

- __state extractor__: The raw memory of breakout is a very large state space, so a script is used to extract key variables from memory (by selecting only the variables that change between frames, which are likely to be a useful part of the game state and not constants). Note that any information about the screen pixels (aka the "frame buffer") must be removed in order for the memory agent to learn strictly from the variables that make up the state, and not the pixel data (which is supposed to be the sole domain of the pixel agent)

- __python breakout__: Implementation of the game "breakout" in python, modified to return the variables that make up the game state as a flat array. This is an "ideal state" representation, since it contains only a minimal selection of variables useful for interpreting the state of the game.

- __GBC breakout__: Emulation of the game "breakout", using a Gameboy Color ROM.

- __GBC render__: Given a dump of internal memory, return the frame buffer (screen pixels) for that state



[X] Find a breakout implementation online
- https://github.com/aknuck/Atari-Breakout/

[X] Create __python breakout__: Implement breakout and add a method to return variables that make up the game state as a flat array

[ ] Create __memory agent__ for __python breakout__: Train memory agent with PPO to play __python breakout__ using the "ideal state" provided by the custom breakout implementation

[ ] Create __pixel agent__ for __python breakout__: Train RL model with PPO to play breakout using downscaled, greyscale screen pixels as its state

[ ] Compare performance (loss, and time taken to converge) of __memory agent__ against __pixel agent__ on __python breakout__

[ ] Create __GBC breakout__: Use pyboy to emulate breakout on GameBoy Color, and add a method to access the internal memory / RAM. The game should start on the first frame of the core gameplay (skip the title screen) and end when the player loses their last ball

[ ] Create __state extractor__: Write a script to extract useful variables from emulator memory by tracking which variables change between frames (i.e. remove constants)

[ ] Create __memory agent__ for __GBC breakout__: Train RL model with PPO to play breakout using the extracted state

[ ] Create __pixel agent__ for __GBC breakout__: Train RL model with PPO to play breakout using downscaled, greyscale screen pixels from emulated breakout

[ ] Compare performance (loss, and time taken to converge) of __memory agent__ against __pixel agent__ on __GBC breakout__

[ ] Create __GBC render__

[ ] Once __memory agent__ has identified key variables in __GBC breakout__ RAM, identify function vectors in the __pixel agent__ by randomly sampling these variables, and generating the corresponding screen pixels before feeding them into __pixel agent__ and measuring activations

[ ] Visualise all activations in __pixel agent__, labelled with the variable (RAM address) they represent

[ ] Repeat the __GBC breakout__ experiment across a wider variety of Gameboy Color games and extract features for all games

[ ] Use the __pixel_agent__'s function vectors as a guide to which RAM addresses map to which parts of the screen, and annotate activations chart with pictures of the screen