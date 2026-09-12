# Security

## Reporting

Please report a vulnerability through GitHub's private advisory form on this repository
("Security" → "Report a vulnerability"), not in a public issue. I aim to reply within a week.

## What this tool does with your machine

- It reads the **index** of model files, locally or over HTTPS. It does not download weights and
  does not execute anything from a model file.
- `moefit bench` writes a 2 GB test file under the directory you point it at, reads it back and
  leaves it in place for reuse. Delete `.moefit-read-test` to reclaim the space.
- `moefit verify` runs the `llama.cpp` binary **you** name, with arguments printed beforehand.
- Nothing is sent anywhere. There is no telemetry and no network access beyond the model URL you
  pass in.
