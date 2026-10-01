# Cross-platform public error privacy

Issue #114 tracks the Windows baseline failures. This focused branch repairs
only partial filesystem-path disclosure in media diagnostics and serialized
stem-separation errors. It starts from main
`72b47ceafa96f14c896f037bb883e5d1f49c87c9`, independently of score PR #113.

The reproduction uses synthetic path strings. Replacing a configured root
before recognizing its complete descendant strips the drive prefix, leaving
private directory and file names that no longer match the Windows-path filter.
Public diagnostics must redact the whole path, including descendants, while
retaining bounded useful failure and recovery text. Credentials and traceback
handling must remain intact.

No source, database, model, lifecycle, API schema, or product-definition change
is made. The remaining resource-import, symlink-privilege, readiness-cleanup
and worker-classification failures remain separately tracked in #114.

The shared string sanitizer matches whole configured paths (longest first),
including descendants and mixed separators, before a replacement can hide the
path grammar. Quoted paths preserve punctuation-rich filenames as one unit;
generic Windows drive, UNC, and POSIX forms are then redacted. Windows known
paths are matched case-insensitively. A following drive prefix in a multi-path
message is not consumed by the first match. Existing separation credential,
URL, traceback and output-size policies remain in force.

Unquoted path/prose boundaries can be ambiguous, particularly when names
contain spaces, commas or semicolons. Those characters do not end a path;
the filter may conservatively remove adjacent diagnostic text up to a colon;
it is not a reversible path parser. Original technical errors remain in private
server logs, while public recovery controls and earlier artifacts are unchanged.

Validation:

```text
python -X utf8 -m pytest tests/test_media.py tests/test_public_error_redaction.py tests/test_stem_api.py
pytest
python -m compileall -q app tests
node --check app/static/app.js
```
