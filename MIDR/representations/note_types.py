""" 
    Note Classes 
"""

import numpy as np
import matplotlib.pyplot as plt



class StdNote:
    def __init__(self, onset, offset, pitch, velocity):
        if velocity < 0 or velocity > 127:
            raise ValueError(f"Velocity must be between 0 and 127")
        
        self.onset    = onset
        self.offset   = offset
        self.pitch    = pitch
        self.velocity = velocity

    def __repr__(self):
        return str(f"Midi Note:({self.__dict__})")
    
    def to_dict(self):
        return {
            "onset"    : float(self.onset),
            "offset"   : float(self.offset),
            "pitch"    : self.pitch,
            "velocity" : self.velocity
        }
    
    @classmethod
    def from_midi_note(cls, midi_note):
        onset    = midi_note["onset"]
        offset   = midi_note["offset"]
        pitch    = midi_note["pitch"]
        velocity = midi_note["velocity"]

        return cls(onset, offset, pitch, velocity)
    
    @classmethod
    def from_other_note_rep(cls, note):
        assert(isinstance(note, SnaNote) or 
               isinstance(note, TorusNote))

        onset    = note.onset
        offset   = note.offset
        pitch    = note.pitch
        velocity = note.velocity

        return cls(onset, offset, pitch, velocity)



class SnaNote:
    def __init__(self, onset, offset, pitch, velocity):
        if velocity < 0 or velocity > 127:
            raise ValueError(f"Velocity must be between 0 and 127")
        
        self.onset    = onset
        self.offset   = offset
        self.pitch    = pitch
        self.velocity = velocity

        A = np.sqrt(2.0 / 15.0) * np.pi / 2.0
        R = 1.0
        T = 12 * A # ~ 6.883

        # helix logic
    
    def __repr__(self):
        return str(f"Helix Note:({self.__dict__})")
    
    def to_dict(self):
        return {
            "onset"    : float(self.onset),
            "offset"   : float(self.offset),
            "pitch"    : self.pitch,
            "velocity" : self.velocity
        }

    def visualize(self):
        return 0
    
    def to_midi_note(self):
        return StdNote(self.onset,
                        self.offset,
                        self.pitch,
                        self.velocity)

    @classmethod
    def from_midi_note(cls, midi_note):
        # assert(isinstance(midi_note, MidiNote))

        start    = midi_note.onset
        end      = midi_note.offset
        pitch    = midi_note.pitch
        velocity = midi_note.velocity

        return cls(start, end, pitch, velocity)

    

class TorusNote:
    def __init__(self, onset, offset, pitch, velocity):
        if velocity < 0 or velocity > 127:
            raise ValueError(f"Velocity must be between 0 and 127")
        
        self.onset    = onset
        self.offset   = offset
        self.pitch    = pitch
        self.velocity = velocity

        R = 1
        r = 0.5

        # Create logic
        theta = np.pi / 2
        phi   = np.pi
        self.x = (R + r * np.cos(phi)) * np.cos(theta)
        self.y = (R + r * np.cos(phi)) * np.sin(theta)
        self.z = r * np.sin(phi)

    def __repr__(self):
        return str(f"Torus Note:({self.__dict__})")
    
    def to_dict(self):
        return {
            "onset"    : float(self.onset),
            "offset"   : float(self.offset),
            "pitch"    : self.pitch,
            "velocity" : self.velocity
        }

    def visualize(self):
        R = 1
        r = 0.5

        theta = np.linspace(0, 2 * np.pi, 12)
        phi   = np.linspace(0, 2 * np.pi, 12)
        theta, phi = np.meshgrid(theta, phi)

        # Torus surface
        X = (R + r * np.cos(phi)) * np.cos(theta)
        Y = (R + r * np.cos(phi)) * np.sin(theta)
        Z = r * np.sin(phi)

        # Plot
        fig = plt.figure(figsize=(8, 6))
        ax  = fig.add_subplot(111, projection='3d')

        # Plot torus surface
        ax.plot_surface(X, Y, Z, rstride=2, cstride=2, color='lightblue', alpha=0.5, edgecolor='k', linewidth=0.3)

        # Plot point
        ax.scatter([self.x], [self.y], [self.z], color='black', s=100, label='Note')

        # Labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        # Equal aspect ratio
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        plt.show()

    def to_midi_note(self):
        return StdNote(self.onset,
                        self.offset,
                        self.pitch,
                        self.velocity)
    
    @classmethod
    def from_midi_note(cls, midi_note):
        start    = midi_note.onset
        end      = midi_note.offset
        pitch    = midi_note.pitch
        velocity = midi_note.velocity

        return cls(start, end, pitch, velocity)



if __name__ == "__main__":
    note = TorusNote(0, 1, 64, 64)
    note.visualize()