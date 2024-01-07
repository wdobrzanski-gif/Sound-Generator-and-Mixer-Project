"""
Module: song_generator

Module with functions for PSA #4 of COMP 110.

Authors:
1) Will Dobrzanski - wdobrzanski@sandiego.edu
2) Brian Manriquez - bmanrique@sandiego.edu
"""

import sound

# Do NOT modify the scale_volume function
def scale_volume(original_sound, factor):
    """
    Decreases the volume of a sound object by a specified factor.

    Paramters:
    original_sound (type; Sound): The sound object whose volume is to be decreased.
    factor (type: float): The factor by which the volume is to be decreased.

    Returns:
    (type: Sound) A new sound object that is a copy of original_sound, but with volumes
    scaled by factor.
    """

    scaled_sound = sound.copy(original_sound)

    for smpl in scaled_sound:
        # Scale left channel of smpl
        current_left = smpl.left
        scaled_left = round(current_left * factor)
        smpl.left = scaled_left

        # Scale right channel of smpl
        current_right = smpl.right
        scaled_right = round(current_right * factor)
        smpl.right = scaled_right

    return scaled_sound


def mix_sounds(snd1, snd2):
    """
    Mixes together two sounds (snd1 and snd2) into a single sound.
    If the sounds are of different length, the mixed sound will be the length
    of the longer sound.

    This returns a new sound: it does not modify either of the original
    sounds.

    Parameters:
    snd1 (type: Sound) - The first sound to mix
    snd2 (type: Sound) - The second sound to mix

    Returns:
    (type: Sound) A Sound object that combines the two parameter sounds into a
    single, overlapping sound.
    """

    if len(snd1) > len(snd2):
        shortest = len(snd2)
        newsound = sound.copy(snd1)

    else:
        shortest = len(snd1)
        newsound = sound.copy(snd2)

    for lit in range(0, shortest):
        newsound[lit].left = snd1[lit].left + snd2[lit].left
        newsound[lit].right = snd1[lit].right + snd2[lit].right

    return newsound



def song_generator(notestring):
    """
    Generates a sound object containing a song specified by the notestring.

    Parameter:
    notestring (type: string) - A string of musical notes and characters to
    change the volume and/or octave of the song.

    Returns:
    (type: Sound) A song generated from the notestring given as a paramter.
    """
    sd = sound.create_silent_sound(1)
    length = 7350 * 2
    mul = 0
    oct = 0
    volmult = 1
    first = None

    #2nd implementation
    if notestring[0] == "|":
        last = notestring.find("|")
        bpm = int(notestring[1:last])
        bps = bpm/60
        length = round(44100//bps)
        print(length)

    for i in range(len(notestring)):
        ch = notestring[i]
        if mul > 0:
            real = length * mul
            mul = 0
        else:
            real = length

        if ch in "ABCDEFG":
            sd = sd + scale_volume(sound.Note(ch, real, oct), volmult)
        elif ch == "P":
            sd = sd + scale_volume(sound.create_silent_sound(real), volmult)
        elif ch.isdigit():
            mul = int(ch)
        elif ch == ">":
            oct = oct + 1
        elif ch == "<":                
            oct = oct - 1
        elif ch == "+":
            volmult = volmult + 0.2
        elif ch == "-":
            volmult = volmult - 0.2
        elif ch == "|":
            first = sd
            sd = sound.create_silent_sound(1)
        if first is not None:
            sd = mix_sounds(first, sd)
    return sd




"""
Don't modify anything below this point.
"""

def main():
    """
    Asks the user for a notestring, generates the song from that
    notestring, then plays the resulting song.
    """
    import sounddevice
    print("Enter a notestring (without quotes):")
    ns = input()
    song = song_generator(ns)
    song.play()
    sounddevice.wait()

if __name__ == "__main__":
    main()
