qwen2 0.5B

drop table if exists dwm_migrate.closing_verified_time;
create table if not exists dwm_migrate.closing_verified_time
PROPERTIES("replication_num" = "3")
as

select
fuse_policy_code,
endorsement_code,
max(event_time) as verified_time
from dwm_migrate.fact_verified_policies_wt
group by fuse_policy_code, endorsement_code
;


drop table if exists dwm_migrate.paid_verified_time;
create table if not exists dwm_migrate.paid_verified_time
PROPERTIES("replication_num" = "3")
as

select
fuse_policy_code,
payment_id,
endorsement_code,
max(event_time) as verified_time
from dwm_migrate.fact_verified_policies_wt
group by fuse_policy_code, endorsement_code, payment_id
;