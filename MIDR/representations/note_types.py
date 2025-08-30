""" 
    Note Classes 
"""

import numpy as np
import matplotlib.pyplot as plt


class MidiNote:
    def __init__(self, onset, offset, pitch, velocity, delta=0):
        if velocity < 0 or velocity > 127:
            raise ValueError(f"Invalid veolcity: {velocity}, velocity must be between 0 and 127.")
        
        if pitch not in set(range(0, 128)) | {-64, -66, -67}:
            raise ValueError(f"Invalid pitch: {pitch}, pitch must be from 0 to 127 or -64, -66, -67.")

        self.onset    = onset
        self.offset   = offset
        self.pitch    = pitch + delta
        self.velocity = velocity

    def __repr__(self):
        return str(f"Midi Note:({self.to_dict()})")
    
    def to_dict(self):
        return {
            "onset"     : float(self.onset),
            "offset"    : float(self.offset),
            "pitch"     : int(self.pitch),
            "velocity"  : int(self.velocity)    
        }


class MidiTransforms():
    def __init__(self, note, repr_type="standard", delta=0):
        if not isinstance(note, MidiNote) and note["onset"] is None:
            raise TypeError(f"Midi notes must be of type MidiNote or a similarly structured object.")
        
        self.onset      = note.onset
        self.offset     = note.offset
        self.pitch      = note.pitch + delta
        self.velocity   = note.velocity
        self.repr_type  = repr_type
        
        if self.repr_type == "standard":
            self.get_std_coords()
        elif self.repr_type == "sna":
            self.get_sna_coords()
        elif self.repr_type == "torus":
            self.get_torus_coords()
        else:
            raise ValueError(f"Invalid repr type: {self.repr_type}")


    def __repr__(self):
        return str(f"{self.repr_type} note: ({self.to_dict()})")

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
    

    def get_std_coords(self):
        self.x = 0
        self.y = 0
        self.z = self.pitch / 12

    def get_sna_coords(self):
        A = np.sqrt(2.0 / 15.0) * np.pi / 2.0
        R = 1.0

        CIRCLE_OF_FIFTHS_IDX = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

        self.idx = CIRCLE_OF_FIFTHS_IDX.index(self.pitch % 12)

        theta = (np.pi / 6) * self.idx
        
        self.x = R * np.cos(theta)
        self.y = R * np.sin(theta)
        self.z = A * (self.pitch - 24)

    def get_torus_coords(self):
        R = 1
        r = 0.5

        # Torus logic
        theta = np.pi / 2
        phi   = np.pi
        self.x = (R + r * np.cos(phi)) * np.cos(theta)
        self.y = (R + r * np.cos(phi)) * np.sin(theta)
        self.z = r * np.sin(phi)


    def visualize(self):
        if self.repr_type == "standard":
            self.visualize_std()
        elif self.repr_type == "sna":
            self.visualize_sna()
        elif self.repr_type == "torus":
            self.visualize_torus()

    def visualize_std(self):
        return 0
    
    def __generate_helix(self, octaves=8, r=1.0, h=np.sqrt(2.0 / 15.0) * np.pi / 2.0):
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
    
    def visualize_sna(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Helix
        x_helix, y_helix, z_helix = self.__generate_helix(octaves=8)
        ax.plot(x_helix, y_helix, z_helix, color='gray', alpha=0.5, linewidth=1, label="SNA")

        # Note
        ax.scatter(self.x, self.y, self.z, color='black', label=self.pitch)

        ax.set_title(f"SNA rep of pitch {self.pitch}", fontsize=14)
        ax.legend()
    
        plt.tight_layout()
        plt.show()  

    def visualize_torus(self):
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
        if self.repr_type == "standard":
            midi_note = self.std_to_midi_note()
        elif self.repr_type == "sna":
            midi_note = self.sna_to_midi_note()
        elif self.repr_type == "torus":
            midi_note = self.torus_to_midi_note()

    def std_to_midi_note(self):
        return MidiNote(self.onset, self.offset, self.pitch, self.velocity) # Temp
    
    def sna_to_midi_note(self):
        A = np.sqrt(2.0 / 15.0) * np.pi / 2.0
        pitch = int((self.z / A) + 12 * 2)

        return MidiNote(self.onset, self.offset, pitch, self.velocity)
    
    def torus_to_midi_note(self):
        return MidiNote(self.onset, self.offset, self.pitch, self.velocity) # Temp



if __name__ == "__main__":
    sna_note = TorusNote(0, 1, 62, 64, 0)
    sna_note.visualize()
    # sna_note = sna_note.to_dict()

    # print(sna_note)

    # midi_note = StdNote.from_sna_note(sna_note)
    # print(midi_note)