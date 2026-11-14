# Data flow and privacy boundaries

| Component | Stored or transmitted data |
| --- | --- |
| Offline demo | Generated pixels, a fictional label, fake vectors, and local events |
| Enrollment | Display names and face embeddings stored in SQLite |
| Recognition events | Time, type, optional identity, snapshots, and optional clips |
| Web UI / RTSP | Media crosses the configured local network connection |
| SMTP (disabled by default) | Event text and snapshot leave the device for the mail server |
| Optional presence probe | ICMP traffic to an explicitly configured IP address |

The face adapter checks for a local ONNX pack before initializing InsightFace and
disables ONNX Runtime telemetry. This is not an audit of every optional native
dependency. No model weights are downloaded by the demo or the missing-model path.

## Runtime controls

- Bind defaults to `127.0.0.1`. An empty passphrase is allowed only with an explicitly
  local bind. Set a strong `EDGE_CAMERA_WEB_PIN` even on loopback before forwarding
  or proxying the service. A passphrase alone does not encrypt HTTP traffic.
- For remote access, use an authenticated private network and HTTPS, with
  `EDGE_CAMERA_WEB_COOKIE_SECURE=true`. There is no public deployment configuration.
- Each session uses a signed, HTTP-only, same-site cookie. Logout revokes all sessions
  in this single-owner prototype and terminates existing streams on their next check.
- Enrollment accepts at most ten files, 8 MiB per file, 32 MiB per request, and
  20 million pixels per image. Only single-face enrollment is accepted.
- Event retention defaults to 30 days and runs periodically while the service is
  running. `0` explicitly disables it. Files outside the managed event store and
  external mailboxes are not covered.
- Deleting a gallery person removes their templates. Historical event names,
  snapshots, and clips remain until event retention removes them.
- SQLite, photos, and clips are **not encrypted at rest**. Unix file permissions
  are best effort; Windows ACLs and backups require separate administration.
- `.gitignore` excludes runtime data, photos, weights, credentials, and logs.
  Ignore rules are not a secret scanner and do not remove previously tracked files.

The dependency-name test only rejects a list of cloud SDK declarations. It is not
proof that no data can leave the machine. The test suite also blocks outbound socket
connections, but this does not certify the real camera/SMTP/native-runtime paths.

Only use recordings and enrollment images you are authorized to process. This
repository's public preview contains generated shapes, not biometric data.
