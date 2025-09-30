# Disaster recovery drill

- Status: READY
- Generated: 2025-09-29T15:14:46.681264+00:00
- Backup status: OK
- Failover status: OK

## Recovery steps
| Step | Status | Summary |
| --- | --- | --- |
| Backup validation | READY | Backups meet policy targets |
| Failover automation | READY | Failover drill completed |

## Step details
### Backup validation
- Providers: aws_s3, gcs
- Storage: s3://emp-backups/timescale
- Latest backup: 2025-09-29T12:44:46.681264+00:00
- Next due: 2025-09-29T18:44:46.681264+00:00

### Failover automation
- health: fail – health status error
- failover: ok – daily_bars produced zero rows; daily_bars missing symbols: SPY, QQQ, ES; daily_bars status=error
- fallback: ok – fallback executed
- Fallback executed during drill

## Backup readiness snapshot
**Backup readiness – timescale_failover_backups**
- Status: ok
- Generated: 2025-09-29T15:14:46.681264+00:00
- Retention days: 14
- Latest backup: 2025-09-29T12:44:46.681264+00:00
- Next backup due: 2025-09-29T18:44:46.681264+00:00
- Providers: aws_s3, gcs
- Storage: s3://emp-backups/timescale

## Failover drill snapshot
| Component | Status | Summary |
| --- | --- | --- |
| health | FAIL | health status error |
| failover | OK | daily_bars produced zero rows; daily_bars missing symbols: SPY, QQQ, ES; daily_bars status=error |
| fallback | OK | fallback executed |