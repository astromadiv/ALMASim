"""Build and rewrite WSClean command lines (MADIV-28 T4.3).

A WSClean command is ``program options... ms``: the measurement set is the
last argument. :class:`WSCleanCommand` keeps the three apart, so rewriting a
command (another name, another data column, a second pass) never depends on
where in the list an option sits.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

# How many values each option this module removes or replaces takes.
_ARITY = {
    "-abs-mem": 1,
    "-auto-mask": 1,
    "-auto-threshold": 1,
    "-beam-shape": 3,
    "-continue": 0,
    "-data-column": 1,
    "-fits-mask": 1,
    "-j": 1,
    "-make-psf": 0,
    "-mem": 1,
    "-name": 1,
    "-niter": 1,
    "-no-update-model-required": 0,
    "-spws": 1,
    "-temp-dir": 1,
    "-update-model-required": 0,
}
_MODEL_FLAGS = ("-update-model-required", "-no-update-model-required")


@dataclass(frozen=True)
class WSCleanCommand:
    """One WSClean invocation: the program, its options and the measurement set."""

    program: str
    options: tuple[str, ...]
    ms: str

    @classmethod
    def parse(cls, argv: Sequence[object]) -> WSCleanCommand:
        """The command ``argv`` (program first, measurement set last)."""
        parts = [str(part) for part in argv]
        if len(parts) < 2:
            raise ValueError(f"a WSClean command needs a program and an MS: {parts}")
        return cls(parts[0], tuple(parts[1:-1]), parts[-1])

    def argv(self) -> list[str]:
        """The command as an argument list for ``subprocess``."""
        return [self.program, *self.options, self.ms]

    def value(self, flag: str) -> str | None:
        """The value given to ``flag``, or None when it is absent."""
        options = self.options
        for index, part in enumerate(options[:-1]):
            if part == flag:
                return options[index + 1]
        return None

    def without(self, *flags: str) -> WSCleanCommand:
        """The command without ``flags`` and their values."""
        unknown = [flag for flag in flags if flag not in _ARITY]
        if unknown:
            raise ValueError(f"unknown number of values for {unknown}")
        kept: list[str] = []
        options = self.options
        index = 0
        while index < len(options):
            if options[index] in flags:
                index += 1 + _ARITY[options[index]]
                continue
            kept.append(options[index])
            index += 1
        return replace(self, options=tuple(kept))

    def with_option(self, flag: str, *values: object) -> WSCleanCommand:
        """``flag`` set to ``values``: in place when present, otherwise appended."""
        if _ARITY.get(flag, len(values)) != len(values):
            raise ValueError(f"{flag} takes {_ARITY[flag]} value(s), got {len(values)}")
        new = (flag, *(str(value) for value in values))
        options = list(self.options)
        if flag in options:
            at = options.index(flag)
            options[at : at + len(new)] = new
            return replace(self, options=tuple(options))
        return replace(self, options=(*options, *new))

    def with_model_update(self, update: bool) -> WSCleanCommand:
        """Whether WSClean writes the model visibilities (MODEL_DATA) back to the MS."""
        wanted, other = _MODEL_FLAGS if update else _MODEL_FLAGS[::-1]
        if other in self.options:
            options = tuple(wanted if part == other else part for part in self.options)
            return replace(self, options=options)
        return self.with_option(wanted)

    def continued(self) -> WSCleanCommand:
        """The command continuing from the previous run's images (``-continue``)."""
        return self.with_option("-continue")

    def substituted(self, placeholder: str, value: object) -> WSCleanCommand:
        """Every argument equal to ``placeholder`` (an option value or the MS) set to ``value``."""
        options = tuple(str(value) if part == placeholder else part for part in self.options)
        ms = str(value) if self.ms == placeholder else self.ms
        return replace(self, options=options, ms=ms)


def _selfcal_command(cmd: Sequence[str], out: Path, name: str) -> list[str]:
    """The continuum command imaging into ``out/name`` and writing MODEL_DATA."""
    command = WSCleanCommand.parse(cmd).with_option("-name", out / name)
    return command.with_model_update(True).argv()


def _dirty_command(cmd: Sequence[str], out: Path, name: str, column: str) -> list[str]:
    """``cmd`` turned into a dirty image of ``column`` into ``out/name``: same grid,
    weighting and channels, no cleaning, no PSF, the model column left alone."""
    kept = WSCleanCommand.parse(cmd).without(
        "-auto-mask",
        "-auto-threshold",
        "-fits-mask",
        "-data-column",
        "-name",
        "-niter",
        "-continue",
        "-make-psf",
        *_MODEL_FLAGS,
    )
    options = ("-name", str(out / name), *kept.options)
    dirty = replace(kept, options=options).with_option("-niter", 0)
    return dirty.with_model_update(False).with_option("-data-column", column).argv()


def _with_data_column(cmd: Sequence[str], column: str) -> list[str]:
    """``cmd`` reading ``column``."""
    return WSCleanCommand.parse(cmd).with_option("-data-column", column).argv()


def continuum_restart_commands(cmd: Sequence[str]) -> list[list[str]]:
    """The continuum ``cmd`` as two passes: one writing MODEL_DATA, then ``-continue``.

    WSClean's auto-mask is built in the first major cycle, at ``-auto-mask``
    times the noise it measures then; a bright source's sidelobes inflate that
    noise, so the mask misses its fainter parts and later cycles, which clean
    only inside it, never reach them. uid___A002_X10ded83_X6af4 SPW 16: noise
    91 mJy at mask creation against 26 mJy at the end, a 167 mJy feature 0.7
    beam off the peak left uncleaned, 16 % of the flux within a beam in the
    residual. The second pass starts from the true residual, measures the
    noise again and builds a new mask: residual 3 %, model flux +20 %
    (2026-10-18). The model visibilities go into the task's own copy of the MS.
    """
    first = WSCleanCommand.parse(cmd).with_model_update(True)
    return [first.argv(), first.continued().argv()]
