# ROUND0_INVALID_UNCOMMITTED_RUN

Produced from a dirty working tree. HEAD was c743c56, whose round0.py has no executable
main(); round0.py, instance_gen.py and test_round0.py were all modified at the time, and
Addendum C had not entered the prereg. The summary nevertheless recorded c743c56 as the
executing commit, so the provenance is wrong.

Every artifact and hash is kept. The numbers in it — 180 passing units, 6 non-empty
layers, 12 anchors, 36 expanded instances — do not enter any verdict. No criterion was
changed as a result of having seen them; the code and the data are deterministic, so a
re-run on the frozen commit is expected to reproduce the data content bit for bit.
