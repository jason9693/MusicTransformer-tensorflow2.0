# import numpy as np
# import copy, itertools, collections
# from pretty_midi import PrettyMIDI, Note, Instrument
#
# # ==================================================================================
# # Parameters
# # ==================================================================================
#
# # NoteSeq -------------------------------------------------------------------------
#
# DEFAULT_SAVING_PROGRAM = 1
# DEFAULT_LOADING_PROGRAMS = range(128)
# DEFAULT_RESOLUTION = 220
# DEFAULT_TEMPO = 120
# DEFAULT_VELOCITY = 64
# # DEFAULT_PITCH_RANGE = range(21, 109)
# DEFAULT_PITCH_RANGE = range(0, 128)
# DEFAULT_VELOCITY_RANGE = range(21, 109)
# DEFAULT_NORMALIZATION_BASELINE = 60  # C4
#
# # EventSeq ------------------------------------------------------------------------
#
# USE_VELOCITY = True
# BEAT_LENGTH = 60 / DEFAULT_TEMPO
# # DEFAULT_TIME_SHIFT_BINS = 1.15 ** (np.arange(100) / 100)
# # DEFAULT_TIME_SHIFT_BINS = 1.15 ** np.arange(100) / (1.15 ** 99)
# DEFAULT_TIME_SHIFT_BINS = np.arange(1,101) / 100
#
#
# DEFAULT_VELOCITY_STEPS = 32
# DEFAULT_NOTE_LENGTH = BEAT_LENGTH * 2
# MIN_NOTE_LENGTH = BEAT_LENGTH / 2
#
# # ControlSeq ----------------------------------------------------------------------
#
# DEFAULT_WINDOW_SIZE = BEAT_LENGTH * 4
# DEFAULT_NOTE_DENSITY_BINS = np.arange(12) * 3 + 1
#
#
# # ==================================================================================
# # Notes
# # ==================================================================================
#
# class NoteSeq:
#
#     @staticmethod
#     def from_midi(midi, programs=DEFAULT_LOADING_PROGRAMS):
#         notes = itertools.chain(*[
#             inst.notes for inst in midi.instruments
#             if inst.program in programs and not inst.is_drum])
#         return NoteSeq(list(notes))
#
#     @staticmethod
#     def from_midi_file(path, *kargs, **kwargs):
#         midi = PrettyMIDI(path)
#         return NoteSeq.from_midi(midi, *kargs, **kwargs)
#
#     @staticmethod
#     def merge(*note_seqs):
#         notes = itertools.chain(*[seq.notes for seq in note_seqs])
#         return NoteSeq(list(notes))
#
#     def __init__(self, notes=[]):
#         self.notes = []
#         if notes:
#             for note in notes:
#                 assert isinstance(note, Note)
#             notes = filter(lambda note: note.end >= note.start, notes)
#             self.add_notes(list(notes))
#
#     def copy(self):
#         return copy.deepcopy(self)
#
#     def to_midi(self, program=DEFAULT_SAVING_PROGRAM,
#                 resolution=DEFAULT_RESOLUTION, tempo=DEFAULT_TEMPO):
#         midi = PrettyMIDI(resolution=resolution, initial_tempo=tempo)
#         inst = Instrument(program, False, 'NoteSeq')
#         inst.notes = copy.deepcopy(self.notes)
#         midi.instruments.append(inst)
#         return midi
#
#     def to_midi_file(self, path, *kargs, **kwargs):
#         self.to_midi(*kargs, **kwargs).write(path)
#
#     def add_notes(self, notes):
#         self.notes += notes
#         self.notes.sort(key=lambda note: note.start)
#
#     def adjust_pitches(self, offset):
#         for note in self.notes:
#             pitch = note.pitch + offset
#             pitch = 0 if pitch < 0 else pitch
#             pitch = 127 if pitch > 127 else pitch
#             note.pitch = pitch
#
#     def adjust_velocities(self, offset):
#         for note in self.notes:
#             velocity = note.velocity + offset
#             velocity = 0 if velocity < 0 else velocity
#             velocity = 127 if velocity > 127 else velocity
#             note.velocity = velocity
#
#     def adjust_time(self, offset):
#         for note in self.notes:
#             note.start += offset
#             note.end += offset
#
#     def trim_overlapped_notes(self, min_interval=0):
#         last_notes = {}
#         for i, note in enumerate(self.notes):
#             if note.pitch in last_notes:
#                 last_note = last_notes[note.pitch]
#                 if note.start - last_note.start <= min_interval:
#                     last_note.end = max(note.end, last_note.end)
#                     last_note.velocity = max(note.velocity, last_note.velocity)
#                     del self.notes[i]
#                 elif note.start < last_note.end:
#                     last_note.end = note.start
#             else:
#                 last_notes[note.pitch] = note
#
#
# # ==================================================================================
# # Events
# # ==================================================================================
#
# class Event:
#
#     def __init__(self, type, time, value):
#         self.type = type
#         self.time = time
#         self.value = value
#
#     def __repr__(self):
#         return 'Event(type={}, time={}, value={})'.format(
#             self.type, self.time, self.value)
#
#
# class EventSeq:
#     pitch_range = DEFAULT_PITCH_RANGE
#     velocity_range = DEFAULT_VELOCITY_RANGE
#     velocity_steps = DEFAULT_VELOCITY_STEPS
#     time_shift_bins = DEFAULT_TIME_SHIFT_BINS
#
#     @staticmethod
#     def from_note_seq(note_seq):
#         note_events = []
#
#         if USE_VELOCITY:
#             velocity_bins = EventSeq.get_velocity_bins()
#
#         for note in note_seq.notes:
#             if note.pitch in EventSeq.pitch_range:
#                 if USE_VELOCITY:
#                     velocity = note.velocity
#                     velocity = max(velocity, EventSeq.velocity_range.start)
#                     velocity = min(velocity, EventSeq.velocity_range.stop - 1)
#                     velocity_index = np.searchsorted(velocity_bins, velocity)
#                     note_events.append(Event('velocity', note.start, velocity_index))
#
#                 pitch_index = note.pitch - EventSeq.pitch_range.start
#                 note_events.append(Event('note_on', note.start, pitch_index))
#                 note_events.append(Event('note_off', note.end, pitch_index))
#
#         note_events.sort(key=lambda event: event.time)  # stable
#         events = []
#         sphere = 0
#
#         for i, event in enumerate(note_events):
#             events.append(event)
#
#             if event is note_events[-1]:
#                 break
#
#             interval = note_events[i + 1].time - event.time
#             shift = 0
#
#             while interval - shift >= EventSeq.time_shift_bins[0]:
#                 index = np.searchsorted(EventSeq.time_shift_bins,
#                                         interval - shift, side='right') - 1
#                 events.append(Event('time_shift', event.time + shift - sphere, index))
#                 shift += EventSeq.time_shift_bins[index]
#             # note_events[i+1].time -= (interval-shift)
#             sphere += (interval-shift)
#
#         return EventSeq(events)
#
#     @staticmethod
#     def from_array(event_indeces):
#         time = 0
#         events = []
#         for event_index in event_indeces:
#             for event_type, feat_range in EventSeq.feat_ranges().items():
#                 if feat_range.start <= event_index < feat_range.stop:
#                     event_value = event_index - feat_range.start
#                     events.append(Event(event_type, time, event_value))
#                     if event_type == 'time_shift':
#                         time += EventSeq.time_shift_bins[event_value]
#                     break
#
#         return EventSeq(events)
#
#     @staticmethod
#     def dim():
#         return sum(EventSeq.feat_dims().values())
#
#     @staticmethod
#     def feat_dims():
#         feat_dims = collections.OrderedDict()
#         feat_dims['time_shift'] = len(EventSeq.time_shift_bins)
#         feat_dims['note_on'] = len(EventSeq.pitch_range)
#         feat_dims['note_off'] = len(EventSeq.pitch_range)
#         if USE_VELOCITY:
#             feat_dims['velocity'] = EventSeq.velocity_steps
#         return feat_dims
#
#     @staticmethod
#     def feat_ranges():
#         offset = 0
#         feat_ranges = collections.OrderedDict()
#         for feat_name, feat_dim in EventSeq.feat_dims().items():
#             feat_ranges[feat_name] = range(offset, offset + feat_dim)
#             offset += feat_dim
#         return feat_ranges
#
#     @staticmethod
#     def get_velocity_bins():
#         n = EventSeq.velocity_range.stop - EventSeq.velocity_range.start
#         return np.arange(
#             EventSeq.velocity_range.start,
#             EventSeq.velocity_range.stop,
#             n / (EventSeq.velocity_steps - 1))
#
#     def __init__(self, events=[]):
#         for event in events:
#             assert isinstance(event, Event)
#
#         self.events = copy.deepcopy(events)
#
#         # compute event times again
#         time = 0
#         for event in self.events:
#             event.time = time
#             if event.type == 'time_shift':
#                 time += EventSeq.time_shift_bins[event.value]
#
#     def to_note_seq(self):
#         time = 0
#         notes = []
#
#         velocity = DEFAULT_VELOCITY
#         velocity_bins = EventSeq.get_velocity_bins()
#
#         last_notes = {}
#
#         for event in self.events:
#             if event.type == 'note_on':
#                 pitch = event.value + EventSeq.pitch_range.start
#                 note = Note(velocity, pitch, time, None)
#                 notes.append(note)
#                 last_notes[pitch] = note
#
#             elif event.type == 'note_off':
#                 pitch = event.value + EventSeq.pitch_range.start
#
#                 if pitch in last_notes:
#                     note = last_notes[pitch]
#                     note.end = max(time, note.start + MIN_NOTE_LENGTH)
#                     del last_notes[pitch]
#
#             elif event.type == 'velocity':
#                 index = min(event.value, velocity_bins.size - 1)
#                 velocity = velocity_bins[index]
#
#             elif event.type == 'time_shift':
#                 time += EventSeq.time_shift_bins[event.value]
#
#         for note in notes:
#             if note.end is None:
#                 note.end = note.start + DEFAULT_NOTE_LENGTH
#
#             note.velocity = int(note.velocity)
#
#         return NoteSeq(notes)
#
#     def to_array(self):
#         feat_idxs = EventSeq.feat_ranges()
#         idxs = [feat_idxs[event.type][event.value] for event in self.events]
#         dtype = np.uint8 if EventSeq.dim() <= 256 else np.uint16
#         return np.array(idxs, dtype=dtype)
#
#
# # ==================================================================================
# # Controls
# # ==================================================================================
#
# class Control:
#
#     def __init__(self, pitch_histogram, note_density):
#         self.pitch_histogram = pitch_histogram  # list
#         self.note_density = note_density  # int
#
#     def __repr__(self):
#         return 'Control(pitch_histogram={}, note_density={})'.format(
#             self.pitch_histogram, self.note_density)
#
#     def to_array(self):
#         feat_dims = ControlSeq.feat_dims()
#         ndens = np.zeros([feat_dims['note_density']])
#         ndens[self.note_density] = 1.  # [dens_dim]
#         phist = np.array(self.pitch_histogram)  # [hist_dim]
#         return np.concatenate([ndens, phist], 0)  # [dens_dim + hist_dim]
#
#
# class ControlSeq:
#     note_density_bins = DEFAULT_NOTE_DENSITY_BINS
#     window_size = DEFAULT_WINDOW_SIZE
#
#     @staticmethod
#     def from_event_seq(event_seq):
#         events = list(event_seq.events)
#         start, end = 0, 0
#
#         pitch_count = np.zeros([12])
#         note_count = 0
#
#         controls = []
#
#         def _rel_pitch(pitch):
#             return (pitch - 24) % 12
#
#         for i, event in enumerate(events):
#
#             while start < i:
#                 if events[start].type == 'note_on':
#                     abs_pitch = events[start].value + EventSeq.pitch_range.start
#                     rel_pitch = _rel_pitch(abs_pitch)
#                     pitch_count[rel_pitch] -= 1.
#                     note_count -= 1.
#                 start += 1
#
#             while end < len(events):
#                 if events[end].time - event.time > ControlSeq.window_size:
#                     break
#                 if events[end].type == 'note_on':
#                     abs_pitch = events[end].value + EventSeq.pitch_range.start
#                     rel_pitch = _rel_pitch(abs_pitch)
#                     pitch_count[rel_pitch] += 1.
#                     note_count += 1.
#                 end += 1
#
#             pitch_histogram = (
#                 pitch_count / note_count
#                 if note_count
#                 else np.ones([12]) / 12
#             ).tolist()
#
#             note_density = max(np.searchsorted(
#                 ControlSeq.note_density_bins,
#                 note_count, side='right') - 1, 0)
#
#             controls.append(Control(pitch_histogram, note_density))
#
#         return ControlSeq(controls)
#
#     @staticmethod
#     def dim():
#         return sum(ControlSeq.feat_dims().values())
#
#     @staticmethod
#     def feat_dims():
#         note_density_dim = len(ControlSeq.note_density_bins)
#         return collections.OrderedDict([
#             ('pitch_histogram', 12),
#             ('note_density', note_density_dim)
#         ])
#
#     @staticmethod
#     def feat_ranges():
#         offset = 0
#         feat_ranges = collections.OrderedDict()
#         for feat_name, feat_dim in ControlSeq.feat_dims().items():
#             feat_ranges[feat_name] = range(offset, offset + feat_dim)
#             offset += feat_dim
#         return feat_ranges
#
#     @staticmethod
#     def recover_compressed_array(array):
#         feat_dims = ControlSeq.feat_dims()
#         assert array.shape[1] == 1 + feat_dims['pitch_histogram']
#         ndens = np.zeros([array.shape[0], feat_dims['note_density']])
#         ndens[np.arange(array.shape[0]), array[:, 0]] = 1.  # [steps, dens_dim]
#         phist = array[:, 1:].astype(np.float64) / 255  # [steps, hist_dim]
#         return np.concatenate([ndens, phist], 1)  # [steps, dens_dim + hist_dim]
#
#     def __init__(self, controls):
#         for control in controls:
#             assert isinstance(control, Control)
#         self.controls = copy.deepcopy(controls)
#
#     def to_compressed_array(self):
#         ndens = [control.note_density for control in self.controls]
#         ndens = np.array(ndens, dtype=np.uint8).reshape(-1, 1)
#         phist = [control.pitch_histogram for control in self.controls]
#         phist = (np.array(phist) * 255).astype(np.uint8)
#         return np.concatenate([
#             ndens,  # [steps, 1] density index
#             phist  # [steps, hist_dim] 0-255
#         ], 1)  # [steps, hist_dim + 1]
#
#
# if __name__ == '__main__':
#     import pickle, sys
#
#     path = sys.argv[1] if len(sys.argv) > 1 else 'dataset/midi/balamb.mid'
#
#     print(EventSeq.dim())
#
#     print('Converting MIDI to EventSeq')
#     es = EventSeq.from_note_seq(NoteSeq.from_midi_file(path))
#
#     print(NoteSeq.from_midi_file(path).notes)
#     # print(NoteSeq.from_midi(NoteSeq.from_midi_file(path).to_midi()).notes)
#     # assert NoteSeq.from_midi_file(path) == NoteSeq.from_midi(NoteSeq.from_midi_file(path).to_midi())
#     print('Converting EventSeq to MIDI')
#     print([(i, item) for i, item in enumerate(NoteSeq.from_midi_file(path).notes)])
#     print([(i, item) for i, item in enumerate(EventSeq.from_array(es.to_array()).to_note_seq().notes)])
#     #assert NoteSeq.from_midi_file(path).notes == EventSeq.from_array(es.to_array()).to_note_seq().notes
#     assert (es.to_array() == EventSeq.from_array(es.to_array()).to_array()).all()
#
#     mid = EventSeq.from_array(es.to_array()).to_note_seq().to_midi()
#     print(NoteSeq.from_midi(mid).notes)
#     EventSeq.from_array(es.to_array()).to_note_seq().to_midi_file('test.mid')

import numpy as np
import copy, itertools, collections
from pretty_midi import PrettyMIDI, Note, Instrument

# ==================================================================================
# Parameters
# ==================================================================================

# NoteSeq -------------------------------------------------------------------------

DEFAULT_SAVING_PROGRAM = 1
DEFAULT_LOADING_PROGRAMS = range(128)
DEFAULT_RESOLUTION = 220
DEFAULT_TEMPO = 120
DEFAULT_VELOCITY = 64
DEFAULT_PITCH_RANGE = range(21, 109)  # 109-20 = 89
DEFAULT_VELOCITY_RANGE = range(21, 109)  # 109 - 20 = 89
DEFAULT_NORMALIZATION_BASELINE = 60  # C4

# EventSeq ------------------------------------------------------------------------

USE_VELOCITY = True
BEAT_LENGTH = 60 / DEFAULT_TEMPO
DEFAULT_TIME_SHIFT_BINS = 1.15 ** np.arange(32) / 65
DEFAULT_VELOCITY_STEPS = 32
DEFAULT_NOTE_LENGTH = BEAT_LENGTH * 2
MIN_NOTE_LENGTH = BEAT_LENGTH / 2

# ControlSeq ----------------------------------------------------------------------

DEFAULT_WINDOW_SIZE = BEAT_LENGTH * 4
DEFAULT_NOTE_DENSITY_BINS = np.arange(12) * 3 + 1


# ==================================================================================
# Notes
# ==================================================================================

class NoteSeq:

    @staticmethod
    def from_midi(midi, programs=DEFAULT_LOADING_PROGRAMS):
        notes = itertools.chain(*[
            inst.notes for inst in midi.instruments
            if inst.program in programs and not inst.is_drum])
        return NoteSeq(list(notes))

    @staticmethod
    def from_midi_file(path, *kargs, **kwargs):
        midi = PrettyMIDI(path)
        return NoteSeq.from_midi(midi, *kargs, **kwargs)

    @staticmethod
    def merge(*note_seqs):
        notes = itertools.chain(*[seq.notes for seq in note_seqs])
        return NoteSeq(list(notes))

    def __init__(self, notes=[]):
        self.notes = []
        if notes:
            for note in notes:
                assert isinstance(note, Note)
            notes = filter(lambda note: note.end >= note.start, notes)
            self.add_notes(list(notes))

    def copy(self):
        return copy.deepcopy(self)

    def to_midi(self, program=DEFAULT_SAVING_PROGRAM,
                resolution=DEFAULT_RESOLUTION, tempo=DEFAULT_TEMPO):
        midi = PrettyMIDI(resolution=resolution, initial_tempo=tempo)
        inst = Instrument(program, False, 'NoteSeq')
        inst.notes = copy.deepcopy(self.notes)
        midi.instruments.append(inst)
        return midi

    def to_midi_file(self, path, *kargs, **kwargs):
        self.to_midi(*kargs, **kwargs).write(path)

    def add_notes(self, notes):
        self.notes += notes
        self.notes.sort(key=lambda note: note.start)

    def adjust_pitches(self, offset):
        for note in self.notes:
            pitch = note.pitch + offset
            pitch = 0 if pitch < 0 else pitch
            pitch = 127 if pitch > 127 else pitch
            note.pitch = pitch

    def adjust_velocities(self, offset):
        for note in self.notes:
            velocity = note.velocity + offset
            velocity = 0 if velocity < 0 else velocity
            velocity = 127 if velocity > 127 else velocity
            note.velocity = velocity

    def adjust_time(self, offset):
        for note in self.notes:
            note.start += offset
            note.end += offset

    def trim_overlapped_notes(self, min_interval=0):
        last_notes = {}
        for i, note in enumerate(self.notes):
            if note.pitch in last_notes:
                last_note = last_notes[note.pitch]
                if note.start - last_note.start <= min_interval:
                    last_note.end = max(note.end, last_note.end)
                    last_note.velocity = max(note.velocity, last_note.velocity)
                    del self.notes[i]
                elif note.start < last_note.end:
                    last_note.end = note.start
            else:
                last_notes[note.pitch] = note


# ==================================================================================
# Events
# ==================================================================================

class Event:

    def __init__(self, type, time, value):
        self.type = type
        self.time = time
        self.value = value

    def __repr__(self):
        return 'Event(type={}, time={}, value={})'.format(
            self.type, self.time, self.value)


class EventSeq:
    pitch_range = DEFAULT_PITCH_RANGE
    velocity_range = DEFAULT_VELOCITY_RANGE
    velocity_steps = DEFAULT_VELOCITY_STEPS
    time_shift_bins = DEFAULT_TIME_SHIFT_BINS

    @staticmethod
    def from_note_seq(note_seq):
        note_events = []

        if USE_VELOCITY:
            velocity_bins = EventSeq.get_velocity_bins()

        for note in note_seq.notes:
            if note.pitch in EventSeq.pitch_range:
                if USE_VELOCITY:
                    velocity = note.velocity
                    velocity = max(velocity, EventSeq.velocity_range.start)
                    velocity = min(velocity, EventSeq.velocity_range.stop - 1)
                    velocity_index = np.searchsorted(velocity_bins, velocity)
                    note_events.append(Event('velocity', note.start, velocity_index))

                pitch_index = note.pitch - EventSeq.pitch_range.start
                note_events.append(Event('note_on', note.start, pitch_index))
                note_events.append(Event('note_off', note.end, pitch_index))

        note_events.sort(key=lambda event: event.time)  # stable
        events = []

        for i, event in enumerate(note_events):
            events.append(event)

            if event is note_events[-1]:
                break

            interval = note_events[i + 1].time - event.time
            shift = 0

            while interval - shift >= EventSeq.time_shift_bins[0]:
                index = np.searchsorted(EventSeq.time_shift_bins,
                                        interval - shift, side='right') - 1
                events.append(Event('time_shift', event.time + shift, index))
                shift += EventSeq.time_shift_bins[index]

        return EventSeq(events)

    @staticmethod
    def from_array(event_indeces):
        time = 0
        events = []
        for event_index in event_indeces:
            for event_type, feat_range in EventSeq.feat_ranges().items():
                if feat_range.start <= event_index < feat_range.stop:
                    event_value = event_index - feat_range.start
                    events.append(Event(event_type, time, event_value))
                    if event_type == 'time_shift':
                        time += EventSeq.time_shift_bins[event_value]
                    break

        return EventSeq(events)

    @staticmethod
    def dim():
        return sum(EventSeq.feat_dims().values())

    @staticmethod
    def feat_dims():
        feat_dims = collections.OrderedDict()
        feat_dims['note_on'] = len(EventSeq.pitch_range)
        feat_dims['note_off'] = len(EventSeq.pitch_range)
        if USE_VELOCITY:
            feat_dims['velocity'] = EventSeq.velocity_steps
        feat_dims['time_shift'] = len(EventSeq.time_shift_bins)
        return feat_dims

    @staticmethod
    def feat_ranges():
        offset = 0
        feat_ranges = collections.OrderedDict()
        for feat_name, feat_dim in EventSeq.feat_dims().items():
            feat_ranges[feat_name] = range(offset, offset + feat_dim)
            offset += feat_dim
        return feat_ranges

    @staticmethod
    def get_velocity_bins():
        n = EventSeq.velocity_range.stop - EventSeq.velocity_range.start
        return np.arange(
            EventSeq.velocity_range.start,
            EventSeq.velocity_range.stop,
            n / (EventSeq.velocity_steps - 1))

    def __init__(self, events=[]):
        for event in events:
            assert isinstance(event, Event)

        self.events = copy.deepcopy(events)

        # compute event times again
        time = 0
        for event in self.events:
            event.time = time
            if event.type == 'time_shift':
                time += EventSeq.time_shift_bins[event.value]

    def to_note_seq(self):
        time = 0
        notes = []

        velocity = DEFAULT_VELOCITY
        velocity_bins = EventSeq.get_velocity_bins()

        last_notes = {}

        for event in self.events:
            if event.type == 'note_on':
                pitch = event.value + EventSeq.pitch_range.start
                note = Note(velocity, pitch, time, None)
                notes.append(note)
                last_notes[pitch] = note

            elif event.type == 'note_off':
                pitch = event.value + EventSeq.pitch_range.start

                if pitch in last_notes:
                    note = last_notes[pitch]
                    note.end = max(time, note.start + MIN_NOTE_LENGTH)
                    del last_notes[pitch]

            elif event.type == 'velocity':
                index = min(event.value, velocity_bins.size - 1)
                velocity = velocity_bins[index]

            elif event.type == 'time_shift':
                time += EventSeq.time_shift_bins[event.value]

        for note in notes:
            if note.end is None:
                note.end = note.start + DEFAULT_NOTE_LENGTH

            note.velocity = int(note.velocity)

        return NoteSeq(notes)

    def to_array(self):
        feat_idxs = EventSeq.feat_ranges()
        idxs = [feat_idxs[event.type][event.value] for event in self.events]
        dtype = np.uint8 if EventSeq.dim() <= 256 else np.uint16
        return np.array(idxs, dtype=dtype)


# ==================================================================================
# Controls
# ==================================================================================

class Control:

    def __init__(self, pitch_histogram, note_density):
        self.pitch_histogram = pitch_histogram  # list
        self.note_density = note_density  # int

    def __repr__(self):
        return 'Control(pitch_histogram={}, note_density={})'.format(
            self.pitch_histogram, self.note_density)

    def to_array(self):
        feat_dims = ControlSeq.feat_dims()
        ndens = np.zeros([feat_dims['note_density']])
        ndens[self.note_density] = 1.  # [dens_dim]
        phist = np.array(self.pitch_histogram)  # [hist_dim]
        return np.concatenate([ndens, phist], 0)  # [dens_dim + hist_dim]


class ControlSeq:
    note_density_bins = DEFAULT_NOTE_DENSITY_BINS
    window_size = DEFAULT_WINDOW_SIZE

    @staticmethod
    def from_event_seq(event_seq):
        events = list(event_seq.events)
        start, end = 0, 0

        pitch_count = np.zeros([12])
        note_count = 0

        controls = []

        def _rel_pitch(pitch):
            return (pitch - 24) % 12

        for i, event in enumerate(events):

            while start < i:
                if events[start].type == 'note_on':
                    abs_pitch = events[start].value + EventSeq.pitch_range.start
                    rel_pitch = _rel_pitch(abs_pitch)
                    pitch_count[rel_pitch] -= 1.
                    note_count -= 1.
                start += 1

            while end < len(events):
                if events[end].time - event.time > ControlSeq.window_size:
                    break
                if events[end].type == 'note_on':
                    abs_pitch = events[end].value + EventSeq.pitch_range.start
                    rel_pitch = _rel_pitch(abs_pitch)
                    pitch_count[rel_pitch] += 1.
                    note_count += 1.
                end += 1

            pitch_histogram = (
                pitch_count / note_count
                if note_count
                else np.ones([12]) / 12
            ).tolist()

            note_density = max(np.searchsorted(
                ControlSeq.note_density_bins,
                note_count, side='right') - 1, 0)

            controls.append(Control(pitch_histogram, note_density))

        return ControlSeq(controls)

    @staticmethod
    def dim():
        return sum(ControlSeq.feat_dims().values())

    @staticmethod
    def feat_dims():
        note_density_dim = len(ControlSeq.note_density_bins)
        return collections.OrderedDict([
            ('pitch_histogram', 12),
            ('note_density', note_density_dim)
        ])

    @staticmethod
    def feat_ranges():
        offset = 0
        feat_ranges = collections.OrderedDict()
        for feat_name, feat_dim in ControlSeq.feat_dims().items():
            feat_ranges[feat_name] = range(offset, offset + feat_dim)
            offset += feat_dim
        return feat_ranges

    @staticmethod
    def recover_compressed_array(array):
        feat_dims = ControlSeq.feat_dims()
        assert array.shape[1] == 1 + feat_dims['pitch_histogram']
        ndens = np.zeros([array.shape[0], feat_dims['note_density']])
        ndens[np.arange(array.shape[0]), array[:, 0]] = 1.  # [steps, dens_dim]
        phist = array[:, 1:].astype(np.float64) / 255  # [steps, hist_dim]
        return np.concatenate([ndens, phist], 1)  # [steps, dens_dim + hist_dim]

    def __init__(self, controls):
        for control in controls:
            assert isinstance(control, Control)
        self.controls = copy.deepcopy(controls)

    def to_compressed_array(self):
        ndens = [control.note_density for control in self.controls]
        ndens = np.array(ndens, dtype=np.uint8).reshape(-1, 1)
        phist = [control.pitch_histogram for control in self.controls]
        phist = (np.array(phist) * 255).astype(np.uint8)
        return np.concatenate([
            ndens,  # [steps, 1] density index
            phist  # [steps, hist_dim] 0-255
        ], 1)  # [steps, hist_dim + 1]


if __name__ == '__main__':
    import pickle, sys

    path = sys.argv[1] if len(sys.argv) > 1 else 'dataset/sample/c_maj.mid'

    print(EventSeq.dim())

    print('Converting MIDI to EventSeq')
    es = EventSeq.from_note_seq(NoteSeq.from_midi_file(path))

    print('Converting EventSeq to MIDI')
    EventSeq.from_array(es.to_array()[:30]).to_note_seq().to_midi_file('test.mid')
    print(list(es.to_array()[:30]))

    # print('Converting EventSeq to ControlSeq')
    # cs = ControlSeq.from_event_seq(es)

    # print('Saving compressed ControlSeq')
    # pickle.dump(cs.to_compressed_array(), open('/tmp/cs-compressed.data', 'wb'))
    #
    # print('Loading compressed ControlSeq')
    # c = ControlSeq.recover_compressed_array(pickle.load(open('/tmp/cs-compressed.data', 'rb')))

    print('Done')
