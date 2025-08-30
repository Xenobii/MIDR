""" 
    Note Classes 
"""

import numpy as np
import matplotlib.pyplot as plt



class StdNote:
    def __init__(self, onset, offset, pitch, velocity, delta=0):
        if velocity < 0 or velocity > 127:
            raise ValueError(f"Velocity must be between 0 and 127")
        
        self.onset    = onset
        self.offset   = offset
        self.pitch    = pitch + delta
        self.velocity = velocity

        self.x = 0
        self.y = 0
        self.z = self.pitch / 12

    def __repr__(self):
        return str(f"Midi Note:({self.__dict__})")
    
    def to_dict(self):
        return {
            "onset"     : float(self.onset),
            "offset"    : float(self.offset),
            "pitch"     : int(self.pitch),
            "velocity"  : int(self.velocity),
            "x"         : float(self.x),
            "y"         : float(self.y),
            "z"         : float(self.z)      
        }
    
    @classmethod
    def from_midi_note(cls, midi_note, delta):
        onset    = midi_note["onset"]
        offset   = midi_note["offset"]
        pitch    = midi_note["pitch"]
        velocity = midi_note["velocity"]

        return cls(onset, offset, pitch, velocity, delta)
    
    @classmethod
    def from_sna_note(cls, sna_note, delta=0):
        A = np.sqrt(2.0 / 15.0) * np.pi / 2.0
        z = sna_note["z"]
        onset = sna_note["onset"]
        offset = sna_note["offset"]
        velocity = sna_note["velocity"]

        pitch = int((z / A) + 12 * 2)

        return cls(onset, offset, pitch, velocity, delta)
    
    @classmethod
    def from_torus_note(cls, torus_note, delta=0):
        # temp
        pitch = 0
        onset = torus_note["onset"]
        offset = torus_note["offset"]
        velocity = torus_note["velocity"]

        return cls(onset, offset, pitch, velocity, delta)
    

class SnaNote:
    def __init__(self, onset, offset, pitch, velocity, delta=0):
        if velocity < 0 or velocity > 127:
            raise ValueError(f"Velocity must be between 0 and 127")
        
        self.onset    = onset
        self.offset   = offset
        self.pitch    = pitch + delta
        self.velocity = velocity

        # Helix logic
        A = np.sqrt(2.0 / 15.0) * np.pi / 2.0
        R = 1.0

        CIRCLE_OF_FIFTHS_IDX = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

        self.idx = CIRCLE_OF_FIFTHS_IDX.index(self.pitch % 12)

        theta = (np.pi / 6) * self.idx
        
        self.x = R * np.cos(theta)
        self.y = R * np.sin(theta)
        self.z = A * (self.pitch - 24) 
    
    def __repr__(self):
        return str(f"Helix Note:({self.__dict__})")
    
    def to_dict(self):
        return {
            "onset"     : float(self.onset),
            "offset"    : float(self.offset),
            "pitch"     : int(self.pitch),
            "velocity"  : int(self.velocity),
            "x"         : float(self.x),
            "y"         : float(self.y),
            "z"         : float(self.z)      
        }
    
    def __generate_helix__(self, octaves=8, r=1.0, h=np.sqrt(2.0 / 15.0) * np.pi / 2.0):
        """
        Returns coordinates for a helix array for visualization
        """
        pitch_range = range(0, 12 * octaves)
        x_vals, y_vals, z_vals = [], [], []
        smooth_factor = 1
        for p in pitch_range:
            theta = (np.pi / 6) * p * smooth_factor
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = h * p * smooth_factor
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)
        return np.array(x_vals), np.array(y_vals), np.array(z_vals)

    def visualize(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Helix
        x_helix, y_helix, z_helix = self.__generate_helix__(octaves=8)
        ax.plot(x_helix, y_helix, z_helix, color='gray', alpha=0.5, linewidth=1, label="SNA")

        # Note
        ax.scatter(self.x, self.y, self.z, color='black', label=self.pitch)

        ax.set_title(f"SNA rep of pitch {self.pitch}", fontsize=14)
        ax.legend()
    
        plt.tight_layout()
        plt.show()

    @classmethod
    def from_midi_note(cls, midi_note, delta):
        onset    = midi_note["onset"]
        offset   = midi_note["offset"]
        pitch    = midi_note["pitch"]
        velocity = midi_note["velocity"]

        return cls(onset, offset, pitch, velocity, delta)
    

class TorusNote:
    def __init__(self, onset, offset, pitch, velocity, delta):
        if velocity < 0 or velocity > 127:
            raise ValueError(f"Velocity must be between 0 and 127")
        
        self.onset    = onset
        self.offset   = offset
        self.pitch    = pitch + delta
        self.velocity = velocity

        R = 1
        r = 0.5

        # Torus logic
        theta = np.pi / 2
        phi   = np.pi
        self.x = (R + r * np.cos(phi)) * np.cos(theta)
        self.y = (R + r * np.cos(phi)) * np.sin(theta)
        self.z = r * np.sin(phi)

    def __repr__(self):
        return str(f"Torus Note:({self.__dict__})")
    
    def to_dict(self):
        return {
            "onset"     : float(self.onset),
            "offset"    : float(self.offset),
            "pitch"     : int(self.pitch),
            "velocity"  : int(self.velocity),
            "x"         : float(self.x),
            "y"         : float(self.y),
            "z"         : float(self.z)      
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
    sna_note = TorusNote(0, 1, 62, 64, 0)
    sna_note.visualize()
    # sna_note = sna_note.to_dict()

    # print(sna_note)

    # midi_note = StdNote.from_sna_note(sna_note)
    # print(midi_note)