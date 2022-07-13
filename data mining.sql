#########################################################################################################################
create materialized view tar_p as
with pro as (
    select height_first_day.icustay_id,
           case when weight<=percentile_cont(0.995::double precision) within group (order by weight)
                 and weight>=percentile_cont(0.995::double precision) within group (order by weight)
          then weight
            end
    as weight,
            case when height<=percentile_cont(0.995::double precision) within group (order by height)
                 and height>=percentile_cont(0.995::double precision) within group (order by height)
          then height
            end
    as height
    from public.height_first_day left join public.weight_first_day
    on height_first_day.icustay_id = weight_first_day.icustay_id
    group by height_first_day.icustay_id,weight_first_day.weight,height_first_day.height
),
tar_p as(select icustays.subject_id,
            icustays.hadm_id,
            icustays.icustay_id,
            icustays.intime,
            icustays.outtime,
            patients.dob,
            patients.gender,
            round(date_part('epoch', icustays.outtime - icustays.intime)::NUMERIC/60)AS ZHUYUANTIME,
            case
            when icustays.intime+ '28 days' >= patients.dod then 1 else 0
            end as death_label,
            COALESCE(pro.weight,percentile_cont(0.5) WITHIN GROUP (ORDER BY pro.weight) ) /(COALESCE(pro.height,percentile_cont(0.5) WITHIN GROUP (ORDER BY pro.height) )/100)^2 as bmi,
            case
            when date_part('year'::text, icustays.intime) - date_part('year'::text, patients.dob)=300 then 91.4
            else date_part('year'::text, icustays.intime) - date_part('year'::text, patients.dob) end as age
    from mimiciii.icustays left join mimiciii.patients
    on icustays.subject_id = patients.subject_id
    left join pro
    on icustays.icustay_id = pro.icustay_id
    where (date_part('year'::text, icustays.intime) - date_part('year'::text, patients.dob)) >= 18
    and round(date_part('epoch', outtime - intime)::NUMERIC/60) >= 1440
    and round(date_part('epoch', outtime - intime)::NUMERIC/60)<= 40320
    group by icustays.subject_id,icustays.hadm_id,icustays.icustay_id,icustays.intime,icustays.outtime,patients.dob,gender,pro.weight,pro.height,patients.dod)
select *
from tar_p
############################################################################################################################
create materialized view pivoted_bg_art11 as
WITH pivoted_bg_art11 AS (
    SELECT pivoted_bg_art.icustay_id,
           pivoted_bg_art.hadm_id,
           art1.gender,
		   art1.age,
           art1.bmi,
           floor(date_part('epoch'::text, pivoted_bg_art.charttime - art1.intime) /
                 3600::double precision)                                                                     AS charttime,
           floor(date_part('epoch'::text, art1.outtime - pivoted_bg_art.charttime) / 3600::double precision) AS outtime,
           art1.death_label,
           art1.intime,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg_art.peep)                AS peep,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg_art.so2)                 AS so2,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg_art.pco2)                AS pco2,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg_art.po2)                 AS po2,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg_art.spo2)                AS spo2,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_bg_art.fio2_chartevents)                                           AS fio2_chartevents,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg_art.fio2)                AS fio2,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg_art.aado2)               AS aado2,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_bg_art.aado2_calc)                                                 AS aado2_calc,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_bg_art.pao2fio2ratio)                                              AS pao2fio2ratio,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg_art.ph)                  AS ph,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_bg_art.bicarbonate)                                                AS bicarbonate,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_bg_art.baseexcess)                                                 AS baseexcess,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_bg_art.totalco2)                                                   AS totalco2,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_bg_art.hematocrit)                                                 AS hematocrit,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_bg_art.hemoglobin)                                                 AS hemoglobin,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_bg_art.carboxyhemoglobin)                                          AS carboxyhemoglobin,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_bg_art.methemoglobin)                                              AS methemoglobin,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_bg_art.chloride)                                                   AS chloride,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg_art.calcium)             AS calcium,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_bg_art.temperature)                                                AS temperature,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_bg_art.potassium)                                                  AS potassium,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg_art.sodium)              AS sodium,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg_art.glucose)             AS glucose,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg_art.lactate)             AS lactate,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_bg_art.intubated)                                                  AS intubated,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_bg_art.tidalvolume)                                                AS tidalvolume,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_bg_art.ventilationrate)                                            AS ventilationrate,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_bg_art.ventilator)                                                 AS ventilator,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg_art.o2flow)              AS o2flow,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_bg_art.requiredo2)                                                 AS requiredo2
    FROM (SELECT tar_p.icustay_id,
                 tar_p.hadm_id,
                 tar_p.gender,
				 tar_p.age,
                 tar_p.bmi,
                 tar_p.intime,
                 tar_p.outtime,
                 tar_p.death_label
          FROM tar_p
          GROUP BY tar_p.icustay_id, tar_p.hadm_id, tar_p.gender, tar_p.age, tar_p.bmi, tar_p.intime, tar_p.outtime,
                   tar_p.death_label) art1
             LEFT JOIN mimiciii.pivoted_bg_art
                       ON art1.icustay_id = pivoted_bg_art.icustay_id AND art1.hadm_id = pivoted_bg_art.hadm_id
    GROUP BY pivoted_bg_art.icustay_id, pivoted_bg_art.hadm_id, art1.gender, art1.age, art1.bmi, pivoted_bg_art.charttime,
             art1.intime, art1.outtime, art1.death_label
)
SELECT pivoted_bg_art11.icustay_id,
       pivoted_bg_art11.hadm_id,
       pivoted_bg_art11.gender,
	   pivoted_bg_art11.age,
       pivoted_bg_art11.bmi,
       pivoted_bg_art11.charttime,
       pivoted_bg_art11.outtime,
       pivoted_bg_art11.death_label,
       pivoted_bg_art11.intime,
       pivoted_bg_art11.peep,
       pivoted_bg_art11.so2,
       pivoted_bg_art11.pco2,
       pivoted_bg_art11.po2,
       pivoted_bg_art11.spo2,
       pivoted_bg_art11.fio2_chartevents,
       pivoted_bg_art11.fio2,
       pivoted_bg_art11.aado2,
       pivoted_bg_art11.aado2_calc,
       pivoted_bg_art11.pao2fio2ratio,
       pivoted_bg_art11.ph,
       pivoted_bg_art11.bicarbonate,
       pivoted_bg_art11.baseexcess,
       pivoted_bg_art11.totalco2,
       pivoted_bg_art11.hematocrit,
       pivoted_bg_art11.hemoglobin,
       pivoted_bg_art11.carboxyhemoglobin,
       pivoted_bg_art11.methemoglobin,
       pivoted_bg_art11.chloride,
       pivoted_bg_art11.calcium,
       pivoted_bg_art11.temperature,
       pivoted_bg_art11.potassium,
       pivoted_bg_art11.sodium,
       pivoted_bg_art11.glucose,
       pivoted_bg_art11.lactate,
       pivoted_bg_art11.intubated,
       pivoted_bg_art11.tidalvolume,
       pivoted_bg_art11.ventilationrate,
       pivoted_bg_art11.ventilator,
       pivoted_bg_art11.o2flow,
       pivoted_bg_art11.requiredo2
FROM pivoted_bg_art11
################################################################################################################
create materialized view pivoted_bg11 as
with pivoted_bg11 as (
select pivoted_bg.icustay_id, pivoted_bg.hadm_id, bg1.gender,bg1.age, bg1.bmi, FLOOR(date_part('epoch',pivoted_bg.charttime-bg1.intime)/3600) as charttime,
FLOOR(date_part('epoch',bg1.outtime-pivoted_bg.charttime)/3600) as outtime,
death_label, intime,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.aado2) as aado2,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.baseexcess) as baseexcess,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.bicarbonate) as bicarbonate,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.totalco2)as totalco2,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.carboxyhemoglobin)as carboxyhemoglobin,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.chloride)as chloride,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.calcium)as calcium,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.glucose)as glucose,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.hematocrit)as hematocrit,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.hemoglobin)as hemoglobin,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.intubated)as intubated,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.lactate)as lactate,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.methemoglobin)as methemoglobin,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.o2flow)as o2flow,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.so2)as so2,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.fio2)as fio2,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.pco2)as pco2,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.peep)as peep,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.ph)as ph,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.po2)as po2,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.potassium)as potassium,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.requiredo2)as requiredo2,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.sodium)as sodium,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.temperature)as temperature,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.ventilationrate)as ventilationrate,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.ventilator)as ventilator,
percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_bg.tidalvolume)as tidalvolume 
from (select icustay_id,hadm_id,gender,age,bmi,intime,outtime,death_label
from tar_p
group by icustay_id,hadm_id,gender,age,bmi,intime,outtime,death_label ) as bg1   
left join mimiciii.pivoted_bg
on bg1.icustay_id= pivoted_bg.icustay_id
and bg1.hadm_id= pivoted_bg. hadm_id
group by pivoted_bg.icustay_id, pivoted_bg.hadm_id, bg1.gender, bg1.age, bg1.bmi,pivoted_bg.charttime,bg1.intime,bg1.outtime,death_label)
SELECT pivoted_bg11.icustay_id,
       pivoted_bg11.hadm_id,
       pivoted_bg11.gender,
	   pivoted_bg11.age,
       pivoted_bg11.bmi,
       pivoted_bg11.charttime,
       pivoted_bg11.outtime,
       pivoted_bg11.death_label,
       pivoted_bg11.intime,
       pivoted_bg11.aado2,
       pivoted_bg11.baseexcess,
       pivoted_bg11.bicarbonate,
       pivoted_bg11.totalco2,
       pivoted_bg11.carboxyhemoglobin,
       pivoted_bg11.chloride,
       pivoted_bg11.calcium,
       pivoted_bg11.glucose,
       pivoted_bg11.hematocrit,
       pivoted_bg11.hemoglobin,
       pivoted_bg11.intubated,
       pivoted_bg11.lactate,
       pivoted_bg11.methemoglobin,
       pivoted_bg11.o2flow,
       pivoted_bg11.so2,
       pivoted_bg11.fio2,
       pivoted_bg11.pco2,
       pivoted_bg11.peep,
       pivoted_bg11.ph,
       pivoted_bg11.po2,
       pivoted_bg11.potassium,
       pivoted_bg11.requiredo2,
       pivoted_bg11.sodium,
       pivoted_bg11.temperature,
       pivoted_bg11.ventilationrate,
       pivoted_bg11.ventilator,
       pivoted_bg11.tidalvolume
FROM pivoted_bg11
#######################################################################################################################################
create materialized view pivoted_gcs11 as
WITH pivoted_gcs11 AS (
    SELECT pivoted_gcs.icustay_id,
           gcs1.hadm_id,
           gcs1.gender,
           gcs1.age,
           gcs1.bmi,
           floor(date_part('epoch'::text, pivoted_gcs.charttime - gcs1.intime) / 3600::double precision)  AS charttime,
           floor(date_part('epoch'::text, gcs1.outtime - pivoted_gcs.charttime) / 3600::double precision) AS outtime,
           gcs1.death_label,
           gcs1.intime,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_gcs.gcs)                 AS gcs,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_gcs.gcsmotor)            AS gcsmotor,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_gcs.gcsverbal)           AS gcsverbal,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_gcs.gcseyes)             AS gcseyes,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY (pivoted_gcs.endotrachflag::double precision))                          AS endotrachflag
    FROM (SELECT tar_p.icustay_id,
                 tar_p.hadm_id,
                 tar_p.gender,
	tar_p.age,
                 tar_p.bmi,
                 tar_p.intime,
                 tar_p.outtime,
                 tar_p.death_label
          FROM tar_p
          GROUP BY tar_p.icustay_id, tar_p.hadm_id, tar_p.gender, tar_p.age, tar_p.bmi, tar_p.intime, tar_p.outtime,
                   tar_p.death_label) gcs1
             LEFT JOIN mimiciii.pivoted_gcs ON gcs1.icustay_id = pivoted_gcs.icustay_id
    GROUP BY pivoted_gcs.icustay_id, gcs1.hadm_id, gcs1.gender, gcs1.age, gcs1.bmi, pivoted_gcs.charttime, gcs1.intime,
             gcs1.outtime, gcs1.death_label
)
SELECT pivoted_gcs11.icustay_id,
       pivoted_gcs11.hadm_id,
       pivoted_gcs11.gender,
	   pivoted_gcs11.age,
       pivoted_gcs11.bmi,
       pivoted_gcs11.charttime,
       pivoted_gcs11.outtime,
       pivoted_gcs11.death_label,
       pivoted_gcs11.intime,
       pivoted_gcs11.gcs,
       pivoted_gcs11.gcsmotor,
       pivoted_gcs11.gcsverbal,
       pivoted_gcs11.gcseyes,
       pivoted_gcs11.endotrachflag
FROM pivoted_gcs11
#######################################################################################################################################
create materialized view pivoted_lab11 as
WITH pivoted_lab11 AS (
    SELECT lab1.icustay_id,
           pivoted_lab.hadm_id,
           lab1.gender,
		   lab1.age,
           lab1.bmi,
           floor(date_part('epoch'::text, pivoted_lab.charttime - lab1.intime) / 3600::double precision)  AS charttime,
           floor(date_part('epoch'::text, lab1.outtime - pivoted_lab.charttime) / 3600::double precision) AS outtime,
           lab1.death_label,
           lab1.intime,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_lab.aniongap)            AS aniongap,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_lab.albumin)             AS albumin,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_lab.bands)               AS bands,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_lab.bicarbonate)                                                AS bicarbonate,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_lab.bilirubin)           AS bilirubin,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_lab.creatinine)          AS creatinine,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_lab.chloride)            AS chloride,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_lab.glucose)             AS glucose,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_lab.hematocrit)          AS hematocrit,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_lab.hemoglobin)          AS hemoglobin,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_lab.lactate)             AS lactate,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_lab.platelet)            AS platelet,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_lab.potassium)           AS potassium,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_lab.ptt)                 AS ptt,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_lab.inr)                 AS inr,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_lab.pt)                  AS pt,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_lab.sodium)              AS sodium,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_lab.bun)                 AS bun,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_lab.wbc)                 AS wbc
    FROM (SELECT tar_p.icustay_id,
                 tar_p.hadm_id,
                 tar_p.gender,
				 tar_p.age,
                 tar_p.bmi,
                 tar_p.intime,
                 tar_p.outtime,
                 tar_p.death_label
          FROM tar_p
          GROUP BY tar_p.icustay_id, tar_p.hadm_id, tar_p.gender, tar_p.age, tar_p.bmi, tar_p.intime, tar_p.outtime,
                   tar_p.death_label) lab1
             LEFT JOIN mimiciii.pivoted_lab ON lab1.hadm_id = pivoted_lab.hadm_id
    GROUP BY lab1.icustay_id, pivoted_lab.hadm_id, lab1.gender, lab1.age, lab1.bmi, pivoted_lab.charttime, lab1.intime,
             lab1.outtime, lab1.death_label
)
SELECT pivoted_lab11.icustay_id,
       pivoted_lab11.hadm_id,
       pivoted_lab11.gender,
	   pivoted_lab11.age,
       pivoted_lab11.bmi,
       pivoted_lab11.charttime,
       pivoted_lab11.outtime,
       pivoted_lab11.death_label,
       pivoted_lab11.intime,
       pivoted_lab11.aniongap,
       pivoted_lab11.albumin,
       pivoted_lab11.bands,
       pivoted_lab11.bicarbonate,
       pivoted_lab11.bilirubin,
       pivoted_lab11.creatinine,
       pivoted_lab11.chloride,
       pivoted_lab11.glucose,
       pivoted_lab11.hematocrit,
       pivoted_lab11.hemoglobin,
       pivoted_lab11.lactate,
       pivoted_lab11.platelet,
       pivoted_lab11.potassium,
       pivoted_lab11.ptt,
       pivoted_lab11.inr,
       pivoted_lab11.pt,
       pivoted_lab11.sodium,
       pivoted_lab11.bun,
       pivoted_lab11.wbc
FROM pivoted_lab11
#######################################################################################################################################
create materialized view pivoted_uo11 as
WITH pivoted_uo11 AS (
    SELECT pivoted_uo.icustay_id,
           uo1.hadm_id,
           uo1.gender,
		   uo1.age,
           uo1.bmi,
           floor(date_part('epoch'::text, pivoted_uo.charttime - uo1.intime) / 3600::double precision)  AS charttime,
           floor(date_part('epoch'::text, uo1.outtime - pivoted_uo.charttime) / 3600::double precision) AS outtime,
           uo1.death_label,
           uo1.intime,
           sum(pivoted_uo.urineoutput)                                                                  AS urineoutput
    FROM (SELECT tar_p.icustay_id,
                 tar_p.hadm_id,
                 tar_p.gender,
				 tar_p.age,
                 tar_p.bmi,
                 tar_p.intime,
                 tar_p.outtime,
                 tar_p.death_label
          FROM tar_p
          GROUP BY tar_p.icustay_id, tar_p.hadm_id, tar_p.gender, tar_p.age, tar_p.bmi, tar_p.intime, tar_p.outtime,
                   tar_p.death_label) uo1
             LEFT JOIN mimiciii.pivoted_uo ON uo1.icustay_id = pivoted_uo.icustay_id
    GROUP BY pivoted_uo.icustay_id, uo1.hadm_id, uo1.gender, uo1.age, uo1.bmi, pivoted_uo.charttime, uo1.intime, uo1.outtime,
             uo1.death_label
)
SELECT pivoted_uo11.icustay_id,
       pivoted_uo11.hadm_id,
       pivoted_uo11.gender,
	   pivoted_uo11.age,
       pivoted_uo11.bmi,
       pivoted_uo11.charttime,
       pivoted_uo11.outtime,
       pivoted_uo11.death_label,
       pivoted_uo11.intime,
       pivoted_uo11.urineoutput
FROM pivoted_uo11
#######################################################################################################################################
create materialized view pivoted_vital11 as
WITH pivoted_vital11 AS (
    SELECT pivoted_vital.icustay_id,
           vital1.hadm_id,
           vital1.gender,
		   vital1.age
           vital1.bmi,
           floor(date_part('epoch'::text, pivoted_vital.charttime - vital1.intime) /
                 3600::double precision) AS                                                                      charttime,
           floor(date_part('epoch'::text, vital1.outtime - pivoted_vital.charttime) /
                 3600::double precision) AS                                                                      outtime,
           vital1.death_label,
           vital1.intime,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_vital.heartrate) AS                                                    heartrate,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_vital.sysbp) AS                 sysbp,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_vital.diasbp) AS                diasbp,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_vital.meanbp) AS                meanbp,
           percentile_cont(0.5::double precision)
           WITHIN GROUP (ORDER BY pivoted_vital.resprate) AS                                                     resprate,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_vital.tempc) AS                 tempc,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_vital.spo2) AS                  spo2,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY pivoted_vital.glucose) AS               glucose
    FROM (SELECT tar_p.icustay_id,
                 tar_p.hadm_id,
                 tar_p.gender,
				 tar_p.age,
                 tar_p.bmi,
                 tar_p.intime,
                 tar_p.outtime,
                 tar_p.death_label
          FROM tar_p
          GROUP BY tar_p.icustay_id, tar_p.gender, tar_p.age, tar_p.bmi, tar_p.hadm_id, tar_p.intime, tar_p.outtime,
                   tar_p.death_label) vital1
             LEFT JOIN pivoted_vital ON vital1.icustay_id = pivoted_vital.icustay_id
    GROUP BY pivoted_vital.icustay_id, vital1.hadm_id, vital1.gender, vital11.age, vital1.bmi, pivoted_vital.charttime,
             vital1.intime, vital1.outtime, vital1.death_label
)
SELECT pivoted_vital11.icustay_id,
       pivoted_vital11.hadm_id,
       pivoted_vital11.gender,
	   pivoted_vital11.age,
       pivoted_vital11.bmi,
       pivoted_vital11.charttime,
       pivoted_vital11.outtime,
       pivoted_vital11.death_label,
       pivoted_vital11.intime,
       pivoted_vital11.heartrate,
       pivoted_vital11.sysbp,
       pivoted_vital11.diasbp,
       pivoted_vital11.meanbp,
       pivoted_vital11.resprate,
       pivoted_vital11.tempc,
       pivoted_vital11.spo2,
       pivoted_vital11.glucose
FROM pivoted_vital11
#######################################################################################################################################
create materialized view mean_air_pressure11 as
WITH mean_air_pressure11 AS (
    SELECT map1.icustay_id,
           mean_air_pressure.hadm_id,
           map1.gender,
		   map1.age,
           map1.bmi,
           floor(date_part('epoch'::text, mean_air_pressure.charttime - map1.intime) / 3600::double precision)  AS charttime,
           floor(date_part('epoch'::text, map1.outtime - mean_air_pressure.charttime) / 3600::double precision) AS outtime,
           map1.death_label,
           map1.intime,
           percentile_cont(0.5::double precision) WITHIN GROUP (ORDER BY mean_air_pressure.mean_air_pressure)   AS mean_air_pressure
          --mean_air_pressure.mean_air_pressure
    FROM (SELECT tar_p.icustay_id,
                 tar_p.hadm_id,
                 tar_p.gender,
				 tar_p.age,
                 tar_p.bmi,
                 tar_p.intime,
                 tar_p.outtime,
                 tar_p.death_label
          FROM tar_p
          GROUP BY tar_p.icustay_id, tar_p.hadm_id, tar_p.gender, tar_p.age, tar_p.bmi, tar_p.intime, tar_p.outtime,
                   tar_p.death_label) map1
             LEFT JOIN public.mean_air_pressure ON map1.hadm_id = mean_air_pressure.hadm_id
    and map1.icustay_id = mean_air_pressure.icustay_id
    GROUP BY map1.icustay_id, mean_air_pressure.hadm_id, map1.gender, map1.age, map1.bmi, mean_air_pressure.charttime, map1.intime,
             map1.outtime, map1.death_label,mean_air_pressure.mean_air_pressure
)
SELECT mean_air_pressure11.icustay_id,
       mean_air_pressure11.hadm_id,
       mean_air_pressure11.gender,
	   mean_air_pressure11.age,
       mean_air_pressure11.bmi,
       mean_air_pressure11.charttime,
       mean_air_pressure11.outtime,
       mean_air_pressure11.death_label,
       mean_air_pressure11.intime,
       mean_air_pressure11.mean_air_pressure
FROM mean_air_pressure11
######################################################################################################################
create materialized view public.ventdurations_classification as
WITH co AS (
    SELECT chartevents.subject_id,
           chartevents.hadm_id,
           chartevents.icustay_id,
           chartevents.charttime,
           max(
                   CASE
                       WHEN (chartevents.itemid = ANY (ARRAY [720, 223849])) AND (chartevents.value::text = ANY
                                                                                  (ARRAY ['CMV'::character varying, 'SIMV+PS'::character varying, 'SIMV'::character varying, 'TCPCV'::character varying, 'PSV/SBT'::character varying, 'CMV/ASSIST/AutoFlow'::character varying, 'APRV/Biphasic+ApnVol'::character varying, 'MMV'::character varying, 'APRV'::character varying, 'APRV/Biphasic+ApnPress'::character varying, 'CMV/ASSIST'::character varying, 'SIMV/VOL'::character varying, 'SYNCHRON SLAVE'::character varying, 'CPAP/PSV+Apn TCPL'::character varying, 'SIMV/PSV'::character varying, 'CPAP/PSV+ApnPres'::character varying, 'MMV/PSV/AutoFlow'::character varying, 'SIMV/AutoFlow'::character varying, 'SIMV/PRES'::character varying, 'MMV/PSV'::character varying, 'SIMV/PSV/AutoFlow'::character varying, 'PRVC/SIMV'::character varying, 'MMV/AutoFlow'::character varying, 'VOL/AC'::character varying, 'CMV/AutoFlow'::character varying, 'PRES/AC'::character varying]::text[]))
                           THEN 1
                       WHEN (chartevents.itemid = ANY (ARRAY [720, 223849])) AND (chartevents.value::text = ANY
                                                                                  (ARRAY ['Pressure Control'::character varying, 'Pressure Support'::character varying, 'CPAP'::character varying, 'CPAP+PS'::character varying, 'CPAP/PSV+ApnVol'::character varying, 'PCV+'::character varying, 'PCV+/PSV'::character varying, 'PCV+Assist'::character varying, 'CPAP/PSV'::character varying, 'PRVC/AC'::character varying, 'Apnea Ventilation'::character varying, 'CPAP/PPS'::character varying, 'ASYNCHRON MASTER'::character varying]::text[]))
                           THEN 2
                       WHEN chartevents.value IS NULL THEN 2
                       ELSE 0
                       END) AS vent_status
    FROM mimiciii.chartevents
    WHERE chartevents.itemid = ANY (ARRAY [720, 223849])
    GROUP BY chartevents.subject_id, chartevents.hadm_id, chartevents.icustay_id, chartevents.charttime
)
SELECT co.subject_id,
       co.hadm_id,
       co.icustay_id,
       ventilation_durations.starttime,
       ventilation_durations.endtime,
       vent_status
FROM co
         LEFT JOIN ventilation_durations ON co.icustay_id = ventilation_durations.icustay_id
GROUP BY co.subject_id, co.hadm_id, co.icustay_id, co.charttime, ventilation_durations.starttime,
         ventilation_durations.endtime;
######################################################################################################################
CREATE materialized view public.trachea_cannula_non_and_lab_param_icupat_28days AS
 SELECT combinat.icustay_id,
    combinat.hadm_id,
    combinat.gender,
    combinat.age,
    combinat.bmi,
    combinat.charttime,
    combinat.outtime,
    combinat.intime,
    max(pivoted_bg11.aado2) AS aado2_bg,
    max(pivoted_bg11.baseexcess) AS baseexcess_bg,
    max(pivoted_bg11.bicarbonate) AS bicarbonate_bg,
    max(pivoted_bg11.totalco2) AS totalco2_bg,
    max(pivoted_bg11.carboxyhemoglobin) AS carboxyhemoglobin_bg,
    max(pivoted_bg11.chloride) AS chloride_bg,
    max(pivoted_bg11.calcium) AS calcium_bg,
    max(pivoted_bg11.glucose) AS glucose_bg,
    max(pivoted_bg11.hematocrit) AS hematocrit_bg,
    max(pivoted_bg11.hemoglobin) AS hemoglobin_bg,
    max(pivoted_bg11.intubated) AS intubated_bg,
    max(pivoted_bg11.lactate) AS lactate_bg,
    max(pivoted_bg11.methemoglobin) AS methemoglobin_bg,
    max(pivoted_bg11.o2flow) AS o2flow_bg,
    max(pivoted_bg11.so2) AS so2_bg,
    max(pivoted_bg11.fio2) AS fio2_bg,
    max(pivoted_bg11.pco2) AS pco2_bg,
    max(pivoted_bg11.peep) AS peep_bg,
    max(pivoted_bg11.ph) AS ph_bg,
    max(pivoted_bg11.po2) AS po2_bg,
    max(pivoted_bg11.potassium) AS potassium_bg,
    max(pivoted_bg11.requiredo2) AS requiredo2_bg,
    max(pivoted_bg11.sodium) AS sodium_bg,
    max(pivoted_bg11.temperature) AS temperature_bg,
    max(pivoted_bg11.ventilationrate) AS ventilationrate_bg,
    max(pivoted_bg11.ventilator) AS ventilator_bg,
    max(pivoted_bg11.tidalvolume) AS tidalvolume_bg,
    max(pivoted_bg_art11.peep) AS peep_bgart,
    max(pivoted_bg_art11.so2) AS so2_bgart,
    max(pivoted_bg_art11.pco2) AS pco2_bgart,
    max(pivoted_bg_art11.po2) AS po2_bgart,
    max(pivoted_bg_art11.spo2) AS spo2_bgart,
    max(pivoted_bg_art11.fio2_chartevents) AS fio2_chartevents_bgart,
    max(pivoted_bg_art11.fio2) AS fio2_bgart,
    max(pivoted_bg_art11.aado2) AS aado2_bgart,
    max(pivoted_bg_art11.aado2_calc) AS aado2_calc_bgart,
    max(pivoted_bg_art11.pao2fio2ratio) AS pao2fio2ratio_bgart,
    max(pivoted_bg_art11.ph) AS ph_bgart,
    max(pivoted_bg_art11.bicarbonate) AS bicarbonate_bgart,
    max(pivoted_bg_art11.baseexcess) AS baseexcess_bgart,
    max(pivoted_bg_art11.totalco2) AS totalco2_bgart,
    max(pivoted_bg_art11.hematocrit) AS hematocrit_bgart,
    max(pivoted_bg_art11.hemoglobin) AS hemoglobin_bgart,
    max(pivoted_bg_art11.carboxyhemoglobin) AS carboxyhemoglobin_bgart,
    max(pivoted_bg_art11.methemoglobin) AS methemoglobin_bgart,
    max(pivoted_bg_art11.chloride) AS chloride_bgart,
    max(pivoted_bg_art11.calcium) AS calcium_bgart,
    max(pivoted_bg_art11.temperature) AS temperature_bgart,
    max(pivoted_bg_art11.potassium) AS potassium_bgart,
    max(pivoted_bg_art11.sodium) AS sodium_bgart,
    max(pivoted_bg_art11.glucose) AS glucose_bgart,
    max(pivoted_bg_art11.lactate) AS lactate_bgart,
    max(pivoted_bg_art11.intubated) AS intubated_bgart,
    max(pivoted_bg_art11.tidalvolume) AS tidalvolume_bgart,
    max(pivoted_bg_art11.ventilationrate) AS ventilationrate_bgart,
    max(pivoted_bg_art11.ventilator) AS ventilator_bgart,
    max(pivoted_bg_art11.o2flow) AS o2flow_bgart,
    max(pivoted_bg_art11.requiredo2) AS requiredo2_bgart,
    max(pivoted_gcs11.gcs) AS gcs,
    max(pivoted_gcs11.gcsmotor) AS gcsmotor,
    max(pivoted_gcs11.gcsverbal) AS gcsverbal,
    max(pivoted_gcs11.gcseyes) AS gcseyes,
    max(pivoted_gcs11.endotrachflag) AS endotrachflag,
    max(pivoted_lab11.aniongap) AS aniongap_lab,
    max(pivoted_lab11.albumin) AS albumin_lab,
    max(pivoted_lab11.bands) AS bands_lab,
    max(pivoted_lab11.bicarbonate) AS bicarbonate_lab,
    max(pivoted_lab11.bilirubin) AS bilirubin_lab,
    max(pivoted_lab11.creatinine) AS creatinine_lab,
    max(pivoted_lab11.chloride) AS chloride_lab,
    max(pivoted_lab11.glucose) AS glucose_lab,
    max(pivoted_lab11.hematocrit) AS hematocrit_lab,
    max(pivoted_lab11.hemoglobin) AS hemoglobin_lab,
    max(pivoted_lab11.lactate) AS lactate_lab,
    max(pivoted_lab11.platelet) AS platelet_lab,
    max(pivoted_lab11.potassium) AS potassium_lab,
    max(pivoted_lab11.ptt) AS ptt_lab,
    max(pivoted_lab11.inr) AS inr_lab,
    max(pivoted_lab11.pt) AS pt_lab,
    max(pivoted_lab11.sodium) AS sodium_lab,
    max(pivoted_lab11.bun) AS bun_lab,
    max(pivoted_lab11.wbc) AS wbc_lab,
    max(pivoted_uo11.urineoutput) AS urineoutput,
    max(pivoted_vital11.heartrate) AS heartrate,
    max(pivoted_vital11.sysbp) AS sysbp,
    max(pivoted_vital11.diasbp) AS diasbp,
    max(pivoted_vital11.meanbp) AS meanbp,
    max(pivoted_vital11.resprate) AS resprate,
    max(pivoted_vital11.tempc) AS tempc,
    max(pivoted_vital11.spo2) AS spo2,
    max(pivoted_vital11.glucose) AS glucose,
    max(mean_air_pressure11.mean_air_pressure) AS mean_air_pressure,
    max(
        CASE
            WHEN combinat.charttime >= round(date_part('epoch', ventdurations_classification.starttime - combinat.intime::timestamp)::numeric / 3600)
             AND combinat.charttime <= round(date_part('epoch', ventdurations_classification.endtime - combinat.intime::timestamp)::numeric / 3600)
               THEN ventdurations_classification.vent_status
            ELSE 0
       END) AS vent_statu,
    combinat.death_label,
    combinat.outtime  AS to_discharge
   FROM combinat
     LEFT JOIN ventdurations_classification ON combinat.icustay_id = ventdurations_classification.icustay_id AND combinat.hadm_id = ventdurations_classification.hadm_id
     LEFT JOIN pivoted_bg11 ON combinat.icustay_id = pivoted_bg11.icustay_id AND combinat.hadm_id = pivoted_bg11.hadm_id AND combinat.charttime = pivoted_bg11.charttime
     LEFT JOIN pivoted_bg_art11 ON combinat.icustay_id = pivoted_bg_art11.icustay_id AND combinat.hadm_id = pivoted_bg_art11.hadm_id AND combinat.charttime = pivoted_bg_art11.charttime
     LEFT JOIN pivoted_gcs11 ON combinat.icustay_id = pivoted_gcs11.icustay_id AND combinat.hadm_id = pivoted_gcs11.hadm_id AND combinat.charttime = pivoted_gcs11.charttime
     LEFT JOIN pivoted_uo11 ON combinat.icustay_id = pivoted_uo11.icustay_id AND combinat.hadm_id = pivoted_uo11.hadm_id AND combinat.charttime = pivoted_uo11.charttime
     LEFT JOIN pivoted_vital11 ON combinat.icustay_id = pivoted_vital11.icustay_id AND combinat.hadm_id = pivoted_vital11.hadm_id AND combinat.charttime = pivoted_vital11.charttime
     LEFT JOIN pivoted_lab11 ON combinat.icustay_id = pivoted_lab11.icustay_id AND combinat.hadm_id = pivoted_lab11.hadm_id AND combinat.charttime = pivoted_lab11.charttime
     LEFT JOIN mean_air_pressure11 ON combinat.icustay_id = mean_air_pressure11.icustay_id AND combinat.hadm_id = mean_air_pressure11.hadm_id AND combinat.charttime = mean_air_pressure11.charttime

  WHERE combinat.charttime >= 0
 and combinat.charttime <= 24*28 --人的采样
 and combinat.icustay_id not in (select icustay_id from combinat where combinat.charttime  > 24*28 group by icustay_id) --人
  GROUP BY combinat.icustay_id, combinat.hadm_id, combinat.gender,combinat.age, combinat.bmi, combinat.charttime, combinat.outtime, combinat.intime, combinat.death_label
,ventdurations_classification.starttime
  order BY combinat.icustay_id, combinat.hadm_id, combinat.gender,combinat.age ,combinat.bmi, combinat.charttime, combinat.outtime, combinat.intime, combinat.death_label
,ventdurations_classification.starttime
######################################################################################################################
create materialized view final_dabiao as
select
       t.icustay_id ,
       t.hadm_id ,
       case when (gender = 'M')then 1 else 0 end as gender_1,
       case when (gender = 'M')then 0 else 1 end as gender_0,
       t.age,
       t.bmi  ,
       t.charttime  ,
       case when (mechvent = '1')then 1 else 0 end as mechvent_1,
       case when (mechvent = '1')then 0 else 1 end as mechvent_0,
       case when (oxygentherapy = '1')then 1 else 0 end as oxygentherapy_1,
       case when (oxygentherapy = '1')then 0 else 1 end as oxygentherapy_0,
       t.aado2_bg  ,
       t.baseexcess_bg   ,
       t.bicarbonate_bg  ,
       t.totalco2_bg   ,
       t.carboxyhemoglobin_bg  ,
       t.chloride_bg   ,
       t.calcium_bg  ,
       t.glucose_bg   ,
       t.hematocrit_bg   ,
       t.hemoglobin_bg  ,
       t.intubated_bg  ,
       t.lactate_bg  ,
       t.methemoglobin_bg  ,
       t.o2flow_bg  ,
       t.so2_bg   ,
       t.fio2_bg  ,
       t.pco2_bg   ,
       t.peep_bg  ,
       t.ph_bg  ,
       t.po2_bg   ,
       t.potassium_bg  ,
       t.requiredo2_bg  ,
       t.sodium_bg   ,
       t.temperature_bg  ,
       t.ventilationrate_bg  ,
       t.ventilator_bg  ,
       t.tidalvolume_bg  ,
       t.peep_bgart  ,
       t.so2_bgart   ,
       t.pco2_bgart   ,
       t.po2_bgart   ,
       t.spo2_bgart   ,
       t.fio2_chartevents_bgart   ,
       t.fio2_bgart  ,
       t.aado2_bgart  ,
       t.aado2_calc_bgart  ,
       t.pao2fio2ratio_bgart   ,
       t.ph_bgart  ,
       t.bicarbonate_bgart  ,
       t.baseexcess_bgart   ,
       t.totalco2_bgart   ,
       t.hematocrit_bgart   ,
       t.hemoglobin_bgart  ,
       t.carboxyhemoglobin_bgart  ,
       t.methemoglobin_bgart  ,
       t.chloride_bgart   ,
       t.calcium_bgart  ,
       t.temperature_bgart  ,
       t.potassium_bgart  ,
       t.sodium_bgart   ,
       t.glucose_bgart   ,
       t.lactate_bgart  ,
       t.intubated_bgart  ,
       t.tidalvolume_bgart  ,
       t.ventilationrate_bgart  ,
       t.ventilator_bgart  ,
       t.o2flow_bgart  ,
       t.requiredo2_bgart  ,
       t.gcs   ,
       t.gcsmotor   ,
       t.gcsverbal   ,
       t.gcseyes   ,
       t.endotrachflag   ,
       t.aniongap_lab   ,
       t.albumin_lab  ,
       t.bands_lab  ,
       t.bicarbonate_lab   ,
       t.bilirubin_lab  ,
       t.creatinine_lab  ,
       t.chloride_lab   ,
       t.glucose_lab   ,
       t.hematocrit_lab  ,
       t.hemoglobin_lab  ,
       t.lactate_lab  ,
       t.platelet_lab   ,
       t.potassium_lab  ,
       t.ptt_lab  ,
       t.inr_lab  ,
       t.pt_lab  ,
       t.sodium_lab   ,
       t.bun_lab   ,
       t.wbc_lab  ,
       t.urineoutput   ,
       t.heartrate   ,
       t.sysbp   ,
       t.diasbp   ,
       t.meanbp   ,
       t.resprate  ,
       t.tempc  ,
       t.spo2   ,
       t.glucose   ,
       t.vent_statu   ,
       t.death_label   ,
       t.to_discharge
from (select * from test_trachea_cannula_non_and_lab_param_icupat_28days) as t 
left join
    (select v.icustay_id,
            date_part('epoch', v.charttime - tar_p.intime::timestamp):: NUMERIC / 3600 as charttime,
            v.mechvent,
            v.oxygentherapy
    from ventilation_classification as v
    left join tar_p
        on v.icustay_id=tar_p.icustay_id) as vent 
        on t.icustay_id=vent.icustay_id and t.charttime=vent.charttime
where to_discharge >= 0 




