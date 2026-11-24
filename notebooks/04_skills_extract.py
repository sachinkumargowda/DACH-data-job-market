# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Skill extraction 
# MAGIC
# MAGIC **Read this before you publish anything from this notebook.**
# MAGIC
# MAGIC The Adzuna API truncates descriptions to roughly 450 characters, and
# MAGIC what survives is the company intro, not the requirements list. So
# MAGIC these counts measure *"skills mentioned in the title or the first
# MAGIC paragraph"*, not *"skills required"*.
# MAGIC
# MAGIC That is still a legitimate measurement. It is not the measurement
# MAGIC people will assume you made, so you have to say which one it is,
# MAGIC every single time you show these numbers.
# MAGIC
# MAGIC ## Why a dictionary and not an LLM
# MAGIC 1. **Reproducibility** - identical results on every run.
# MAGIC 2. **Defensibility** - every match points at a specific string.
# MAGIC 3. **Cost** - you have no quota to spare.
# MAGIC
# MAGIC

# COMMAND ----------

CAT    = "jobs"
SILVER = f"{CAT}.silver"

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
import re

# COMMAND ----------

# MAGIC %md ## The dictionary
# MAGIC

# COMMAND ----------

SKILLS = {
 "sql":        {"cat": "language",   "aliases": [r"\bsql\b"]},
 "python":     {"cat": "language",   "aliases": [r"\bpython\b"]},
 "scala":      {"cat": "language",   "aliases": [r"\bscala\b"]},
 "java":       {"cat": "language",   "aliases": [r"\bjava\b(?!script)"]},
 "javascript": {"cat": "language",   "aliases": [r"\bjavascript\b",
                                                 r"\btypescript\b"]},
 "php":        {"cat": "language",   "aliases": [r"\bphp\b"]},
 "r_lang":     {"cat": "language",   "aliases": [
                    r"(?<=[ ,/(])r(?=[ ,/)])"]},


 "spark":      {"cat": "processing", "aliases": [r"\bapache spark\b",
                                                 r"\bpyspark\b",
                                                 r"\bspark\b"]},
 "hadoop":     {"cat": "legacy",     "aliases": [r"\bhadoop\b",
                                                 r"\bhdfs\b",
                                                 r"\bmapreduce\b",
                                                 r"\bhive\b"]},
 "kafka":      {"cat": "streaming",  "aliases": [r"\bkafka\b"]},
 "flink":      {"cat": "streaming",  "aliases": [r"\bflink\b"]},


 "databricks": {"cat": "platform",   "aliases": [r"\bdatabricks\b"]},
 "snowflake":  {"cat": "warehouse",  "aliases": [r"\bsnowflake\b"]},
 "bigquery":   {"cat": "warehouse",  "aliases": [r"\bbigquery\b"]},
 "redshift":   {"cat": "warehouse",  "aliases": [r"\bredshift\b"]},
 "synapse":    {"cat": "warehouse",  "aliases": [r"\bsynapse\b"]},


 "airflow":    {"cat": "orchestr",   "aliases": [r"\bapache airflow\b",
                                                 r"\bairflow\b"]},
 "dbt":        {"cat": "transform",  "aliases": [r"\bdbt\b"]},
 "ssis":       {"cat": "legacy",     "aliases": [r"\bssis\b"]},
 "talend":     {"cat": "legacy",     "aliases": [r"\btalend\b"]},


 "postgres":   {"cat": "database",   "aliases": [r"\bpostgresql\b",
                                                 r"\bpostgres\b"]},
 "mysql":      {"cat": "database",   "aliases": [r"\bmysql\b"]},
 "mssql":      {"cat": "database",   "aliases": [r"\bsql server\b",
                                                 r"\bt-sql\b",
                                                 r"\bms sql\b"]},
 "oracle":     {"cat": "database",   "aliases": [r"\boracle\b"]},
 "mongodb":    {"cat": "database",   "aliases": [r"\bmongodb\b",
                                                 r"\bmongo\b"]},
 "elastic":    {"cat": "database",   "aliases": [r"\belasticsearch\b"]},


 "aws":        {"cat": "cloud",      "aliases": [r"\baws\b",
                                                 r"\bamazon web services\b"]},
 "azure":      {"cat": "cloud",      "aliases": [r"\bazure\b",
                                                 r"\bdata factory\b"]},
 "gcp":        {"cat": "cloud",      "aliases": [r"\bgcp\b",
                                                 r"\bgoogle cloud\b"]},


 "docker":     {"cat": "devops",     "aliases": [r"\bdocker\b"]},
 "kubernetes": {"cat": "devops",     "aliases": [r"\bkubernetes\b",
                                                 r"\bk8s\b"]},
 "terraform":  {"cat": "devops",     "aliases": [r"\bterraform\b"]},
 "git":        {"cat": "devops",     "aliases": [r"\bgithub\b",
                                                 r"\bgitlab\b",
                                                 r"\bgit\b"]},
 "cicd":       {"cat": "devops",     "aliases": [r"\bci/cd\b",
                                                 r"\bjenkins\b"]},


 "powerbi":    {"cat": "bi",         "aliases": [r"\bpower ?bi\b"]},
 "tableau":    {"cat": "bi",         "aliases": [r"\btableau\b"]},
 "looker":     {"cat": "bi",         "aliases": [r"\blooker\b"]},
 "qlik":       {"cat": "bi",         "aliases": [r"\bqlik\b"]},
 "excel":      {"cat": "bi",         "aliases": [r"\bexcel\b"]},


 # --- ML frameworks: how a model gets trained -----------------
 "pytorch":    {"cat": "ml_framework", "aliases": [r"\bpytorch\b"]},
 "tensorflow": {"cat": "ml_framework", "aliases": [r"\btensorflow\b"]},
 "keras":      {"cat": "ml_framework", "aliases": [r"\bkeras\b"]},
 "sklearn":    {"cat": "ml_framework", "aliases": [r"\bscikit-?learn\b",
                                                   r"\bsklearn\b"]},
 "xgboost":    {"cat": "ml_framework", "aliases": [r"\bxgboost\b",
                                                   r"\blightgbm\b",
                                                   r"\bcatboost\b"]},
 "jax":        {"cat": "ml_framework", "aliases": [r"\bjax\b"]},

 # --- MLOps: how a model reaches production -------------------
 # Kept separate from ml_framework because these are the skills
 # that separate "trained a model in a notebook" from "runs a
 # model as a service", and that gap is the whole MLOps story.
 "mlflow":     {"cat": "mlops",      "aliases": [r"\bmlflow\b"]},
 "kubeflow":   {"cat": "mlops",      "aliases": [r"\bkubeflow\b"]},
 "sagemaker":  {"cat": "mlops",      "aliases": [r"\bsagemaker\b",
                                                 r"\bsage maker\b"]},
 "vertexai":   {"cat": "mlops",      "aliases": [r"\bvertex ai\b",
                                                 r"\bvertex\b"]},
 "azureml":    {"cat": "mlops",      "aliases": [r"\bazure ml\b",
                                                 r"\bazure machine "
                                                 r"learning\b",
                                                 r"\bazure ai foundry\b"]},
 "ray":        {"cat": "mlops",      "aliases": [r"\bray serve\b",
                                                 r"\bray\.io\b",
                                                 r"\banyscale\b"]},
 "wandb":      {"cat": "mlops",      "aliases": [r"\bweights ?& ?biases\b",
                                                 r"\bwandb\b"]},
 "bentoml":    {"cat": "mlops",      "aliases": [r"\bbentoml\b",
                                                 r"\bseldon\b",
                                                 r"\btriton\b",
                                                 r"\bkserve\b"]},
 "featurestore": {"cat": "mlops",    "aliases": [r"\bfeature store\b",
                                                 r"\bfeast\b",
                                                 r"\btecton\b"]},

 # --- Generative AI -------------------------------------------
 # "llm" and "rag" used to be aliases of one skill, which made it
 # impossible to see whether DACH employers are asking for models
 # or for retrieval on top of them. They are now separate.
 "llm":        {"cat": "genai",      "aliases": [r"\bllms?\b",
                                                 r"\blarge language "
                                                 r"models?\b",
                                                 r"\bsprachmodell\w*\b",
                                                 r"\bfoundation models?\b"]},
 "genai":      {"cat": "genai",      "aliases": [r"\bgenai\b",
                                                 r"\bgen[\s\-]?ai\b",
                                                 r"\bgenerative ai\b",
                                                 r"\bgenerative ki\b"]},
 "rag":        {"cat": "genai",      "aliases": [r"\bretrieval[\s\-]"
                                                 r"augmented\w*\b",
                                                 r"\brag\b"]},
 "ai_agents":  {"cat": "genai",      "aliases": [r"\bagentic\b",
                                                 r"\bai agents?\b",
                                                 r"\bki[\s\-]agenten?\b",
                                                 r"\bmulti[\s\-]agent\b",
                                                 r"\bmcp\b"]},
 "prompting":  {"cat": "genai",      "aliases": [r"\bprompt[\s\-]?"
                                                 r"engineer\w*\b",
                                                 r"\bprompting\b"]},
 "langchain":  {"cat": "genai",      "aliases": [r"\blangchain\b",
                                                 r"\blanggraph\b",
                                                 r"\bllama[\s\-]?index\b",
                                                 r"\bsemantic kernel\b",
                                                 r"\bhaystack\b"]},
 "huggingface": {"cat": "genai",     "aliases": [r"\bhugging ?face\b",
                                                 r"\btransformers "
                                                 r"library\b"]},
 "openai_api": {"cat": "genai",      "aliases": [r"\bopenai\b",
                                                 r"\bgpt-?[45]\b",
                                                 r"\banthropic\b",
                                                 # not bare "Claude":
                                                 # it is a first name
                                                 # in Swiss postings
                                                 r"\bclaude (api|sonnet|"
                                                 r"opus|code)\b",
                                                 r"\bazure openai\b",
                                                 r"\bmistral\b",
                                                 r"\bgemini\b"]},
 "vectordb":   {"cat": "genai",      "aliases": [r"\bvector (db|"
                                                 r"database|store)\b",
                                                 r"\bvektordatenbank\b",
                                                 r"\bpinecone\b",
                                                 r"\bweaviate\b",
                                                 r"\bqdrant\b",
                                                 r"\bmilvus\b",
                                                 r"\bchroma\b",
                                                 r"\bpgvector\b",
                                                 r"\bfaiss\b"]},
 "finetuning": {"cat": "genai",      "aliases": [r"\bfine[\s\-]?tun\w*\b",
                                                 # LoRaWAN is an IoT
                                                 # radio protocol and
                                                 # shows up in DACH
                                                 # embedded postings
                                                 r"\blora\b(?!wan)",
                                                 r"\bpeft\b",
                                                 r"\brlhf\b",
                                                 r"\bembeddings?\b"]},
 "ai_eval":    {"cat": "genai",      "aliases": [r"\bllm[\s\-]?as[\s\-]?"
                                                 r"a[\s\-]?judge\b",
                                                 r"\bhallucination\w*\b",
                                                 r"\bguardrails?\b",
                                                 r"\bevals?\b",
                                                 r"\bai safety\b",
                                                 r"\bai alignment\b"]},

 # --- NLP and computer vision ---------------------------------
 "nlp":        {"cat": "nlp_cv",     "aliases": [r"\bnlp\b",
                                                 r"\bnatural language "
                                                 r"processing\b",
                                                 r"\bsprachverarbeitung\b",
                                                 r"\bspacy\b",
                                                 r"\bnltk\b"]},
 "cv":         {"cat": "nlp_cv",     "aliases": [r"\bcomputer vision\b",
                                                 r"\bbildverarbeitung\b",
                                                 r"\bbilderkennung\b",
                                                 r"\bopencv\b",
                                                 r"\byolo\b",
                                                 r"\bobject detection\b"]},
 "deeplearning": {"cat": "nlp_cv",   "aliases": [r"\bdeep learning\b",
                                                 r"\bneural network\w*\b",
                                                 r"\bneuronale netze\b",
                                                 r"\btransformer\b",
                                                 r"\bdiffusion model\w*\b"]},


 "sap":        {"cat": "enterprise", "aliases": [r"\bsap\b"]},
 "etl":        {"cat": "practice",   "aliases": [r"\betl\b", r"\belt\b"]},
 "datawarehouse": {"cat": "practice", "aliases": [
                    r"\bdata warehouse\b", r"\bdwh\b",
                    r"\bdata lakehouse\b", r"\bdata lake\b"]},
}


# =====================================================
# ENGINEERING SKILLS EXTENSION
# Personalized DACH Engineering Job Market
# =====================================================

SKILLS.update({

    # =================================================
    # 1. PROGRAMMING LANGUAGES
    # =================================================

    "cpp": {
        "cat": "language",
        "aliases": [r"\bc\+\+", r"\bcpp\b"]
    },

    "c_lang": {
        "cat": "language",
        "aliases": [
            r"\bc programming\b",
            r"\bc language\b",
            r"\bembedded c\b",
            r"\bc-programmierung\b"
        ]
    },

    "csharp": {
        "cat": "language",
        "aliases": [r"\bc#", r"\bcsharp\b"]
    },

    "rust": {
        "cat": "language",
        "aliases": [r"\brust\b"]
    },

    "bash": {
        "cat": "language",
        "aliases": [r"\bbash\b", r"\bshell scripting\b"]
    },

    "cmake": {
        "cat": "devops",
        "aliases": [r"\bcmake\b"]
    },

    # =================================================
    # 2. ROBOTICS AND AUTONOMOUS SYSTEMS
    # =================================================

    "ros2": {
        "cat": "robotics",
        "aliases": [
            r"\bros[\s\-]*2\b",
            r"\brobot operating system 2\b"
        ]
    },

    "ros": {
        "cat": "robotics",
        "aliases": [
            r"\bros\b(?![\s\-]*2)",
            r"\brobot operating system\b(?!\s*2)"
        ]
    },

    "robotics": {
        "cat": "robotics",
        "aliases": [
            r"\brobotics\b",
            r"\brobotik\b",
            r"\brobotersystem\w*\b",
            r"\brobotic systems?\b"
        ]
    },

    "mobile_robotics": {
        "cat": "robotics",
        "aliases": [
            r"\bmobile robotics\b",
            r"\bmobile robots?\b",
            r"\bmobile roboter\b",
            r"\bmobile robotik\b"
        ]
    },

    "slam": {
        "cat": "robotics",
        "aliases": [
            r"\bslam\b",
            r"\bsimultaneous localization and mapping\b",
            r"\bsimultaneous localisation and mapping\b"
        ]
    },

    "localization": {
        "cat": "robotics",
        "aliases": [
            r"\blocali[sz]ation\b",
            r"\blokalisierung\b",
            r"\bpose estimation\b",
            r"\bpositionsbestimmung\b"
        ]
    },

    "mapping": {
        "cat": "robotics",
        "aliases": [
            r"\b3d mapping\b",
            r"\brobot mapping\b",
            r"\boccupancy grid\b",
            r"\bkartierung\b"
        ]
    },

    "path_planning": {
        "cat": "robotics",
        "aliases": [
            r"\bpath planning\b",
            r"\bpathfinding\b",
            r"\bpath finding\b",
            r"\bpfadplanung\b",
            r"\bwegplanung\b",
            r"\broutenplanung\b"
        ]
    },

    "motion_planning": {
        "cat": "robotics",
        "aliases": [
            r"\bmotion planning\b",
            r"\btrajectory planning\b",
            r"\bbewegungsplanung\b",
            r"\btrajektorienplanung\b"
        ]
    },

    "mapf": {
        "cat": "robotics",
        "aliases": [
            r"\bmapf\b",
            r"\bmulti[\s\-]agent path[\s\-]?finding\b"
        ]
    },

    "visual_odometry": {
        "cat": "robotics",
        "aliases": [
            r"\bvisual odometry\b",
            r"\bvisuelle odometrie\b"
        ]
    },

    "vio": {
        "cat": "robotics",
        "aliases": [
            r"\bvio\b",
            r"\bvisual[\s\-]inertial odometry\b",
            r"\bvisuell[\s\-]inertiale odometrie\b"
        ]
    },

    "autonomous_systems": {
        "cat": "robotics",
        "aliases": [
            r"\bautonomous systems?\b",
            r"\bautonome systeme\b",
            r"\bautonomous navigation\b",
            r"\bautonome navigation\b"
        ]
    },

    "amr": {
        "cat": "robotics",
        "aliases": [
            r"\bamrs?\b",
            r"\bautonomous mobile robots?\b"
        ]
    },

    "agv": {
        "cat": "robotics",
        "aliases": [
            r"\bagvs?\b",
            r"\bautomated guided vehicles?\b",
            r"\bfahrerlose transportsysteme\b",
            r"\bfahrerlose transportfahrzeuge\b"
        ]
    },

    "gazebo": {
        "cat": "robotics",
        "aliases": [
            r"\bgazebo\b",
            r"\bignition gazebo\b"
        ]
    },

    "nav2": {
        "cat": "robotics",
        "aliases": [
            r"\bnav2\b",
            r"\bnavigation2\b"
        ]
    },

    "moveit": {
        "cat": "robotics",
        "aliases": [r"\bmoveit(?:\s*2)?\b"]
    },

    "isaac_sim": {
        "cat": "robotics",
        "aliases": [
            r"\bisaac sim\b",
            r"\bisaac lab\b",
            r"\bnvidia isaac\b"
        ]
    },

    "mujoco": {
        "cat": "robotics",
        "aliases": [r"\bmujoco\b"]
    },

    "urdf": {
        "cat": "robotics",
        "aliases": [r"\burdf\b", r"\bxacro\b"]
    },

    "robot_kinematics": {
        "cat": "robotics",
        "aliases": [
            r"\binverse kinematics\b",
            r"\bforward kinematics\b",
            r"\binverse kinematik\b",
            r"\broboterkinematik\b"
        ]
    },

    "humanoid_robotics": {
        "cat": "robotics",
        "aliases": [
            r"\bhumanoid robots?\b",
            r"\bhumanoid robotics\b",
            r"\bhumanoide roboter\b"
        ]
    },

    # =================================================
    # 3. SENSORS AND SENSOR FUSION
    # =================================================

    "lidar": {
        "cat": "sensors",
        "aliases": [
            r"\blidar\b",
            r"\bli[\s\-]dar\b",
            r"\blaser scanners?\b",
            r"\blaserscanner\b"
        ]
    },

    "imu": {
        "cat": "sensors",
        "aliases": [
            r"\bimus?\b",
            r"\binertial measurement units?\b",
            r"\btraegheitsmesseinheit\b",
            r"\bträgheitsmesseinheit\b"
        ]
    },

    "radar": {
        "cat": "sensors",
        "aliases": [
            r"\bradar\b",
            r"\bradarsensor\w*\b"
        ]
    },

    "camera": {
        "cat": "sensors",
        "aliases": [
            r"\bcameras?\b",
            r"\bkameras?\b",
            r"\bkamerasystem\w*\b"
        ]
    },

    "depth_camera": {
        "cat": "sensors",
        "aliases": [
            r"\bdepth cameras?\b",
            r"\brgb[\s\-]?d\b",
            r"\bstereo cameras?\b",
            r"\bstereokamera\b",
            r"\btiefenkamera\b"
        ]
    },

    "ultrasonic": {
        "cat": "sensors",
        "aliases": [
            r"\bultrasonic\b",
            r"\bultrasound sensors?\b",
            r"\bultraschallsensor\w*\b"
        ]
    },

    "gnss": {
        "cat": "sensors",
        "aliases": [
            r"\bgnss\b",
            r"\bgps\b",
            r"\brtk[\s\-]?gps\b",
            r"\brtk[\s\-]?gnss\b"
        ]
    },

    "sensor_fusion": {
        "cat": "sensors",
        "aliases": [
            r"\bsensor[\s\-]?fusion\b",
            r"\bmulti[\s\-]sensor fusion\b",
            r"\bmultisensorfusion\b",
            r"\bsensordatenfusion\b"
        ]
    },

    "sensor_technology": {
        "cat": "sensors",
        "aliases": [
            r"\bsensor technology\b",
            r"\bsensorik\b",
            r"\bsensortechnik\b",
            r"\bsensorentwicklung\b",
            r"\bsensor development\b"
        ]
    },

    "sensor_calibration": {
        "cat": "sensors",
        "aliases": [
            r"\bsensor calibration\b",
            r"\bcamera calibration\b",
            r"\bsensorkalibrierung\b",
            r"\bkamerakalibrierung\b"
        ]
    },

    "time_synchronization": {
        "cat": "sensors",
        "aliases": [
            r"\btime synchronization\b",
            r"\btime synchronisation\b",
            r"\bzeitsynchronisation\b"
        ]
    },

    "signal_processing": {
        "cat": "sensors",
        "aliases": [
            r"\bsignal processing\b",
            r"\bdigital signal processing\b",
            r"\bsignalverarbeitung\b",
            r"\bdigitale signalverarbeitung\b",
            r"\bdsp\b"
        ]
    },

    "cfar": {
        "cat": "sensors",
        "aliases": [
            r"\bcfar\b",
            r"\bconstant false alarm rate\b"
        ]
    },

    # =================================================
    # 4. COMPUTER VISION AND PERCEPTION
    # =================================================

    "opencv": {
        "cat": "perception",
        "aliases": [r"\bopencv\b"]
    },

    "perception": {
        "cat": "perception",
        "aliases": [
            r"\bperception\b",
            r"\bumfeldwahrnehmung\b",
            r"\bumgebungserkennung\b"
        ]
    },

    "pcl": {
        "cat": "perception",
        "aliases": [
            r"\bpcl\b",
            r"\bpoint cloud library\b"
        ]
    },

    "open3d": {
        "cat": "perception",
        "aliases": [r"\bopen3d\b"]
    },

    "point_cloud": {
        "cat": "perception",
        "aliases": [
            r"\bpoint clouds?\b",
            r"\bpunktwolken?\b"
        ]
    },

    "object_detection": {
        "cat": "perception",
        "aliases": [
            r"\bobject detection\b",
            r"\bobjekterkennung\b"
        ]
    },

    "object_tracking": {
        "cat": "perception",
        "aliases": [
            r"\bobject tracking\b",
            r"\bmulti[\s\-]object tracking\b",
            r"\bobjektverfolgung\b"
        ]
    },

    "image_processing": {
        "cat": "perception",
        "aliases": [
            r"\bimage processing\b",
            r"\bdigital image processing\b",
            r"\bindustrielle bildverarbeitung\b"
        ]
    },

    "image_segmentation": {
        "cat": "perception",
        "aliases": [
            r"\bimage segmentation\b",
            r"\bsemantic segmentation\b",
            r"\binstance segmentation\b",
            r"\bbildsegmentierung\b"
        ]
    },

    "yolo": {
        "cat": "perception",
        "aliases": [r"\byolo(?:v?\d+)?\b"]
    },

    "depth_estimation": {
        "cat": "perception",
        "aliases": [
            r"\bdepth estimation\b",
            r"\btiefenschaetzung\b",
            r"\btiefenschätzung\b"
        ]
    },

    "3d_vision": {
        "cat": "perception",
        "aliases": [
            r"\b3d vision\b",
            r"\b3d perception\b",
            r"\b3d reconstruction\b",
            r"\b3d rekonstru\w*\b"
        ]
    },

    # =================================================
    # 5. CONTROL SYSTEMS AND ESTIMATION
    # =================================================

    "control_systems": {
        "cat": "control",
        "aliases": [
            r"\bcontrol systems?\b",
            r"\bcontrol engineering\b",
            r"\bregelungstechnik\b",
            r"\bregelungsentwicklung\b",
            r"\bregelungssystem\w*\b"
        ]
    },

    "pid": {
        "cat": "control",
        "aliases": [
            r"\bpid\b",
            r"\bpid controller\b",
            r"\bpid[\s\-]regler\b"
        ]
    },

    "kalman_filter": {
        "cat": "control",
        "aliases": [
            r"\bkalman filters?\b",
            r"\bkalmanfilter\b",
            r"\bkalman[\s\-]filter\b"
        ]
    },

    "ekf": {
        "cat": "control",
        "aliases": [
            r"\bekf\b",
            r"\bextended kalman filter\b",
            r"\berweiterter kalmanfilter\b"
        ]
    },

    "ukf": {
        "cat": "control",
        "aliases": [
            r"\bukf\b",
            r"\bunscented kalman filter\b"
        ]
    },

    "state_estimation": {
        "cat": "control",
        "aliases": [
            r"\bstate estimation\b",
            r"\bzustandsschaetzung\b",
            r"\bzustandsschätzung\b"
        ]
    },

    "state_space": {
        "cat": "control",
        "aliases": [
            r"\bstate[\s\-]space\b",
            r"\bzustandsraum\b",
            r"\bzustandsraummodell\b"
        ]
    },

    "observer": {
        "cat": "control",
        "aliases": [
            r"\bluenberger observer\b",
            r"\bluenberger[\s\-]beobachter\b",
            r"\bstate observer\b",
            r"\bzustandsbeobachter\b"
        ]
    },

    "mpc": {
        "cat": "control",
        "aliases": [
            r"\bmpc\b",
            r"\bmodel predictive control\b",
            r"\bmodellpraediktive regelung\b",
            r"\bmodellprädiktive regelung\b"
        ]
    },

    "lqr": {
        "cat": "control",
        "aliases": [
            r"\blqr\b",
            r"\blinear quadratic regulator\b"
        ]
    },

    "system_identification": {
        "cat": "control",
        "aliases": [
            r"\bsystem identification\b",
            r"\bsystemidentifikation\b"
        ]
    },

    "dynamic_systems": {
        "cat": "control",
        "aliases": [
            r"\bdynamic systems?\b",
            r"\bdynamische systeme\b",
            r"\bsystem dynamics\b",
            r"\bsystemdynamik\b"
        ]
    },

    # =================================================
    # 6. MATLAB AND MODEL-BASED ENGINEERING
    # =================================================

    "matlab": {
        "cat": "simulation",
        "aliases": [r"\bmatlab\b"]
    },

    "simulink": {
        "cat": "simulation",
        "aliases": [r"\bsimulink\b"]
    },

    "simscape": {
        "cat": "simulation",
        "aliases": [r"\bsimscape\b"]
    },

    "stateflow": {
        "cat": "simulation",
        "aliases": [r"\bstateflow\b"]
    },

    "model_based_development": {
        "cat": "simulation",
        "aliases": [
            r"\bmodel[\s\-]based development\b",
            r"\bmodel[\s\-]based design\b",
            r"\bmodellbasierte entwicklung\b",
            r"\bmodellbasierter entwurf\b",
            r"\bmbd\b"
        ]
    },

    "mbse": {
        "cat": "simulation",
        "aliases": [
            r"\bmbse\b",
            r"\bmodel[\s\-]based systems engineering\b"
        ]
    },

    "multibody": {
        "cat": "simulation",
        "aliases": [
            r"\bmultibody simulation\b",
            r"\bmulti[\s\-]body dynamics\b",
            r"\bmehrkoerpersimulation\b",
            r"\bmehrkörpersimulation\b",
            r"\bmehrkörperdynamik\b"
        ]
    },

    "modelica": {
        "cat": "simulation",
        "aliases": [
            r"\bmodelica\b",
            r"\bopenmodelica\b",
            r"\bdymola\b"
        ]
    },

    "digital_twin": {
        "cat": "simulation",
        "aliases": [
            r"\bdigital twins?\b",
            r"\bdigitaler zwilling\b",
            r"\bdigitale zwillinge\b"
        ]
    },

    "fmi_fmu": {
        "cat": "simulation",
        "aliases": [
            r"\bfmi\b",
            r"\bfmus?\b",
            r"\bfunctional mock[\s\-]up interface\b"
        ]
    },

    "co_simulation": {
        "cat": "simulation",
        "aliases": [
            r"\bco[\s\-]?simulation\b"
        ]
    },

    "hil": {
        "cat": "simulation",
        "aliases": [
            r"\bhil\b",
            r"\bhardware[\s\-]in[\s\-]the[\s\-]loop\b"
        ]
    },

    "sil": {
        "cat": "simulation",
        "aliases": [
            r"\bsil\b",
            r"\bsoftware[\s\-]in[\s\-]the[\s\-]loop\b"
        ]
    },

    "mil": {
        "cat": "simulation",
        "aliases": [
            r"\bmil\b",
            r"\bmodel[\s\-]in[\s\-]the[\s\-]loop\b"
        ]
    },

    "dspace": {
        "cat": "simulation",
        "aliases": [r"\bdspace\b"]
    },

    "virtual_validation": {
        "cat": "simulation",
        "aliases": [
            r"\bvirtual validation\b",
            r"\bvirtuelle validierung\b",
            r"\bvirtual testing\b"
        ]
    },

    # =================================================
    # 7. FEM AND NUMERICAL SIMULATION
    # =================================================

    "fem": {
        "cat": "fem",
        "aliases": [
            r"\bfem\b",
            r"\bfinite[\s\-]element(?:e)?[\s\-]method\w*\b",
            r"\bfinite[\s\-]element simulation\b"
        ]
    },

    "fea": {
        "cat": "fem",
        "aliases": [
            r"\bfea\b",
            r"\bfinite[\s\-]element analysis\b"
        ]
    },

    "abaqus": {
        "cat": "fem",
        "aliases": [r"\babaqus\b"]
    },

    "code_aster": {
        "cat": "fem",
        "aliases": [r"\bcode[\s_\-]?aster\b"]
    },

    "salome_meca": {
        "cat": "fem",
        "aliases": [r"\bsalome[\s_\-]?meca\b"]
    },

    "paraview": {
        "cat": "fem",
        "aliases": [
            r"\bparaview\b",
            r"\bparavis\b"
        ]
    },

    "ansys": {
        "cat": "fem",
        "aliases": [r"\bansys\b"]
    },

    "hypermesh": {
        "cat": "fem",
        "aliases": [
            r"\bhypermesh\b",
            r"\bhyperworks\b"
        ]
    },

    "ls_dyna": {
        "cat": "fem",
        "aliases": [r"\bls[\s\-]?dyna\b"]
    },

    "nastran": {
        "cat": "fem",
        "aliases": [r"\bnastran\b"]
    },

    "comsol": {
        "cat": "fem",
        "aliases": [r"\bcomsol\b"]
    },

    "structural_analysis": {
        "cat": "fem",
        "aliases": [
            r"\bstructural analysis\b",
            r"\bstructural simulation\b",
            r"\bstrukturmechanik\b",
            r"\bstrukturanalyse\b",
            r"\bfestigkeitsberechnung\b"
        ]
    },

    "elastoplasticity": {
        "cat": "fem",
        "aliases": [
            r"\belasto[\s\-]?plastic\w*\b",
            r"\belastoplast\w*\b"
        ]
    },

    "nonlinear_fem": {
        "cat": "fem",
        "aliases": [
            r"\bnon[\s\-]?linear fem\b",
            r"\bnon[\s\-]?linear fea\b",
            r"\bnichtlineare fem\b",
            r"\bnichtlineare simulation\b"
        ]
    },

    "mesh_generation": {
        "cat": "fem",
        "aliases": [
            r"\bmesh generation\b",
            r"\bmesh refinement\b",
            r"\bmesh convergence\b",
            r"\bnetzgenerierung\b",
            r"\bvernetzung\b"
        ]
    },

    "cfd": {
        "cat": "fem",
        "aliases": [
            r"\bcfd\b",
            r"\bcomputational fluid dynamics\b",
            r"\bstroemungssimulation\b",
            r"\bströmungssimulation\b"
        ]
    },

    # =================================================
    # 8. CAD AND MECHANICAL DESIGN
    # =================================================

    "solidworks": {
        "cat": "cad",
        "aliases": [r"\bsolid[\s\-]?works\b"]
    },

    "catia": {
        "cat": "cad",
        "aliases": [r"\bcatia\b"]
    },

    "siemens_nx": {
        "cat": "cad",
        "aliases": [r"\bsiemens nx\b"]
    },

    "creo": {
        "cat": "cad",
        "aliases": [
            r"\bptc creo\b",
            r"\bcreo\b"
        ]
    },

    "autocad": {
        "cat": "cad",
        "aliases": [r"\bautocad\b"]
    },

    "cad": {
        "cat": "cad",
        "aliases": [
            r"\bcad\b",
            r"\bcomputer[\s\-]aided design\b",
            r"\b3d konstruktion\b"
        ]
    },

    "cae": {
        "cat": "cad",
        "aliases": [
            r"\bcae\b",
            r"\bcomputer[\s\-]aided engineering\b"
        ]
    },

    # =================================================
    # 9. DRONES AND FLIGHT SYSTEMS
    # =================================================

    "uav": {
        "cat": "drones",
        "aliases": [
            r"\buavs?\b",
            r"\buas\b",
            r"\bunmanned aerial vehicles?\b",
            r"\bunmanned aircraft systems?\b",
            r"\bdrohnen\b",
            r"\bdrones?\b"
        ]
    },

    "px4": {
        "cat": "drones",
        "aliases": [r"\bpx4\b"]
    },

    "ardupilot": {
        "cat": "drones",
        "aliases": [r"\bardupilot\b"]
    },

    "mavlink": {
        "cat": "drones",
        "aliases": [r"\bmavlink\b"]
    },

    "mavsdk": {
        "cat": "drones",
        "aliases": [r"\bmavsdk\b"]
    },

    "qgroundcontrol": {
        "cat": "drones",
        "aliases": [
            r"\bqgroundcontrol\b",
            r"\bqgc\b"
        ]
    },

    "autopilot": {
        "cat": "drones",
        "aliases": [
            r"\bautopilot\b",
            r"\bflight controller\b"
        ]
    },

    "flight_control": {
        "cat": "drones",
        "aliases": [
            r"\bflight control\b",
            r"\bflugregelung\b",
            r"\bflugsteuerung\b"
        ]
    },

    "ndt": {
        "cat": "testing",
        "aliases": [
            r"\bndt\b",
            r"\bnon[\s\-]?destructive testing\b",
            r"\bzerstoerungsfreie pruefung\b",
            r"\bzerstörungsfreie prüfung\b",
            r"\bzfp\b"
        ]
    },

    # =================================================
    # 10. AUTOMOTIVE AND ADAS
    # =================================================

    "adas": {
        "cat": "automotive",
        "aliases": [
            r"\badas\b",
            r"\badvanced driver assistance systems?\b",
            r"\bfahrerassistenzsystem\w*\b"
        ]
    },

    "autonomous_driving": {
        "cat": "automotive",
        "aliases": [
            r"\bautonomous driving\b",
            r"\bautomated driving\b",
            r"\bself[\s\-]driving\b",
            r"\bautonomes fahren\b",
            r"\bautomatisiertes fahren\b"
        ]
    },

    "vehicle_dynamics": {
        "cat": "automotive",
        "aliases": [
            r"\bvehicle dynamics\b",
            r"\bfahrzeugdynamik\b",
            r"\bfahrdynamik\b"
        ]
    },

    "powertrain": {
        "cat": "automotive",
        "aliases": [
            r"\bpowertrain\b",
            r"\bdrivetrain\b",
            r"\bantriebsstrang\b",
            r"\bantriebssystem\w*\b"
        ]
    },

    "electric_vehicles": {
        "cat": "automotive",
        "aliases": [
            r"\belectric vehicles?\b",
            r"\be[\s\-]?mobility\b",
            r"\belektromobilität\b",
            r"\belektromobilitaet\b"
        ]
    },

    "wltp": {
        "cat": "automotive",
        "aliases": [r"\bwltp\b", r"\bwltc\b"]
    },

    "autosar": {
        "cat": "automotive",
        "aliases": [r"\bautosar\b"]
    },

    "can_bus": {
        "cat": "automotive",
        "aliases": [
            r"\bcan[\s\-]?bus\b",
            r"\bcan[\s\-]?fd\b",
            r"\bcontroller area network\b"
        ]
    },

    "canoe": {
        "cat": "automotive",
        "aliases": [
            r"\bvector canoe\b",
            r"\bcanoe\b"
        ]
    },

    "canalyzer": {
        "cat": "automotive",
        "aliases": [r"\bcanalyzer\b"]
    },

    "iso26262": {
        "cat": "automotive",
        "aliases": [r"\biso[\s\-]?26262\b"]
    },

    "functional_safety": {
        "cat": "automotive",
        "aliases": [
            r"\bfunctional safety\b",
            r"\bfunktionale sicherheit\b",
            r"\basil[\s\-]?[abcd]\b"
        ]
    },

    "automotive_spice": {
        "cat": "automotive",
        "aliases": [
            r"\bautomotive spice\b",
            r"\baspice\b"
        ]
    },

    "ecu": {
        "cat": "automotive",
        "aliases": [
            r"\becus?\b",
            r"\belectronic control units?\b"
        ]
    },

    # =================================================
    # 11. MECHATRONICS AND INDUSTRIAL AUTOMATION
    # =================================================

    "mechatronics": {
        "cat": "mechatronics",
        "aliases": [
            r"\bmechatronics\b",
            r"\bmechatronik\b",
            r"\bmechatronic systems?\b",
            r"\bmechatronische systeme\b"
        ]
    },

    "automation": {
        "cat": "automation",
        "aliases": [
            r"\bindustrial automation\b",
            r"\bautomatisierungstechnik\b",
            r"\bautomatisierung\b"
        ]
    },

    "plc": {
        "cat": "automation",
        "aliases": [
            r"\bplc\b",
            r"\bsps\b",
            r"\bprogrammable logic controller\b",
            r"\bspeicherprogrammierbare steuerung\b"
        ]
    },

    "tia_portal": {
        "cat": "automation",
        "aliases": [
            r"\btia portal\b",
            r"\bsimatic\b",
            r"\bstep[\s\-]?7\b",
            r"\bs7[\s\-]?(1200|1500|300|400)\b"
        ]
    },

    "twincat": {
        "cat": "automation",
        "aliases": [
            r"\btwincat\b",
            r"\bbeckhoff\b"
        ]
    },

    "codesys": {
        "cat": "automation",
        "aliases": [r"\bcodesys\b"]
    },

    "scada": {
        "cat": "automation",
        "aliases": [r"\bscada\b"]
    },

    "hmi": {
        "cat": "automation",
        "aliases": [
            r"\bhmi\b",
            r"\bhuman machine interface\b"
        ]
    },

    "ethercat": {
        "cat": "automation",
        "aliases": [r"\bethercat\b"]
    },

    "profinet": {
        "cat": "automation",
        "aliases": [r"\bprofinet\b"]
    },

    "profibus": {
        "cat": "automation",
        "aliases": [r"\bprofibus\b"]
    },

    "opcua": {
        "cat": "automation",
        "aliases": [r"\bopc[\s\-]?ua\b"]
    },

    "industrial_robots": {
        "cat": "automation",
        "aliases": [
            r"\bindustrial robots?\b",
            r"\bindustrieroboter\b",
            r"\broboterprogrammierung\b",
            r"\brobot programming\b"
        ]
    },

    "robot_brands": {
        "cat": "automation",
        "aliases": [
            r"\bkuka\b",
            r"\bfanuc\b",
            r"\babb robots?\b",
            r"\buniversal robots\b",
            r"\byaskawa\b"
        ]
    },

    "actuators": {
        "cat": "mechatronics",
        "aliases": [
            r"\bactuators?\b",
            r"\baktorik\b",
            r"\baktoren\b",
            r"\baktuatoren\b"
        ]
    },

    "servo_drives": {
        "cat": "mechatronics",
        "aliases": [
            r"\bservo motors?\b",
            r"\bservo drives?\b",
            r"\bservomotor\w*\b",
            r"\bservoantrieb\w*\b"
        ]
    },

    "commissioning": {
        "cat": "automation",
        "aliases": [
            r"\bcommissioning\b",
            r"\binbetriebnahme\b"
        ]
    },

    # =================================================
    # 12. EMBEDDED SYSTEMS AND ELECTRONICS
    # =================================================

    "embedded_systems": {
        "cat": "embedded",
        "aliases": [
            r"\bembedded systems?\b",
            r"\bembedded software\b",
            r"\beingebettete systeme\b"
        ]
    },

    "microcontrollers": {
        "cat": "embedded",
        "aliases": [
            r"\bmicrocontrollers?\b",
            r"\bmikrocontroller\b"
        ]
    },

    "stm32": {
        "cat": "embedded",
        "aliases": [r"\bstm32\b"]
    },

    "arduino": {
        "cat": "embedded",
        "aliases": [r"\barduino\b"]
    },

    "raspberry_pi": {
        "cat": "embedded",
        "aliases": [r"\braspberry pi\b"]
    },

    "jetson": {
        "cat": "embedded",
        "aliases": [
            r"\bnvidia jetson\b",
            r"\bjetson\b"
        ]
    },

    "esp32": {
        "cat": "embedded",
        "aliases": [r"\besp32\b"]
    },

    "rtos": {
        "cat": "embedded",
        "aliases": [
            r"\brtos\b",
            r"\bfreertos\b",
            r"\bzephyr rtos\b",
            r"\breal[\s\-]time operating systems?\b"
        ]
    },

    "fpga": {
        "cat": "embedded",
        "aliases": [r"\bfpga\b"]
    },

    "communication_protocols": {
        "cat": "embedded",
        "aliases": [
            r"\buart\b",
            r"\bi2c\b",
            r"\bspi\b",
            r"\brs[\s\-]?485\b"
        ]
    },

    "pcb": {
        "cat": "embedded",
        "aliases": [
            r"\bpcb design\b",
            r"\bprinted circuit board\b",
            r"\bleiterplattenentwicklung\b"
        ]
    },

    # =================================================
    # 13. OPERATING SYSTEMS AND SOFTWARE TOOLS
    # =================================================

    "linux": {
        "cat": "platform",
        "aliases": [
            r"\blinux\b",
            r"\bubuntu\b",
            r"\bdebian\b"
        ]
    },

    "cuda": {
        "cat": "platform",
        "aliases": [r"\bcuda\b"]
    },

    "tensorrt": {
        "cat": "ml_framework",
        "aliases": [r"\btensorrt\b"]
    },

    "onnx": {
        "cat": "ml_framework",
        "aliases": [r"\bonnx\b"]
    },

    "jira": {
        "cat": "practice",
        "aliases": [r"\bjira\b"]
    },

    "confluence": {
        "cat": "practice",
        "aliases": [r"\bconfluence\b"]
    },

    # =================================================
    # 14. AI FOR ROBOTICS
    # =================================================

    "reinforcement_learning": {
        "cat": "robotics_ai",
        "aliases": [
            r"\breinforcement learning\b",
            r"\bdeep reinforcement learning\b",
            r"\bbestaerkendes lernen\b",
            r"\bbestärkendes lernen\b"
        ]
    },

    "imitation_learning": {
        "cat": "robotics_ai",
        "aliases": [
            r"\bimitation learning\b",
            r"\blearning from demonstration\b"
        ]
    },

    "embodied_ai": {
        "cat": "robotics_ai",
        "aliases": [
            r"\bembodied ai\b",
            r"\bphysical ai\b"
        ]
    },

    "vla": {
        "cat": "robotics_ai",
        "aliases": [
            r"\bvla\b",
            r"\bvision[\s\-]language[\s\-]action\b"
        ]
    },

    "robot_learning": {
        "cat": "robotics_ai",
        "aliases": [
            r"\brobot learning\b",
            r"\brobotic learning\b"
        ]
    },

    "sim_to_real": {
        "cat": "robotics_ai",
        "aliases": [
            r"\bsim[\s\-]to[\s\-]real\b",
            r"\bsim2real\b"
        ]
    },

    # =================================================
    # 15. TESTING AND VALIDATION
    # =================================================

    "verification_validation": {
        "cat": "testing",
        "aliases": [
            r"\bverification and validation\b",
            r"\bverification & validation\b",
            r"\bv&v\b",
            r"\bverifikation und validierung\b"
        ]
    },

    "test_automation": {
        "cat": "testing",
        "aliases": [
            r"\btest automation\b",
            r"\btestautomatisierung\b",
            r"\bautomated testing\b"
        ]
    },

    "system_testing": {
        "cat": "testing",
        "aliases": [
            r"\bsystem testing\b",
            r"\bsystemtests?\b",
            r"\bintegration testing\b",
            r"\bintegrationstests?\b"
        ]
    },

    "fmea": {
        "cat": "testing",
        "aliases": [r"\bfmea\b"]
    },

    "root_cause_analysis": {
        "cat": "testing",
        "aliases": [
            r"\broot cause analysis\b",
            r"\bursachenanalyse\b",
            r"\bfehlerursachenanalyse\b"
        ]
    },

})

# =====================================================
# UPDATE EXISTING COMPUTER VISION DEFINITION
# Prevent overlapping matches with the new skills.
# =====================================================

SKILLS["cv"] = {
    "cat": "perception",
    "aliases": [
        r"\bcomputer vision\b",
        r"\bbildverarbeitung\b",
        r"\bbilderkennung\b",
        r"\bmachine vision\b"
    ]
}

# =====================================================
# GENERATE CATEGORY MAPPING
# Keep this line after all skill definitions.
# =====================================================

CATEGORY = {k: v["cat"] for k, v in SKILLS.items()}


# COMMAND ----------

# MAGIC %md ## The matcher
# MAGIC Longest alias first, then blank the matched span, so `sql` cannot
# MAGIC re-match inside `postgresql`.

# COMMAND ----------

PATTERNS = sorted(
    ((s, a) for s, d in SKILLS.items() for a in d["aliases"]),
    key=lambda p: -len(p[1]))

COMPILED = [(s, re.compile(a, re.I)) for s, a in PATTERNS]

# "R" alone is far too noisy in German text (R&D, R. Mueller,
# street names). Only accept it when the document looks like it
# is listing technologies.
R_CONTEXT = re.compile(
    r"(python|sql|statist|analys|sprachen|languages|matlab|sas)",
    re.I)


def extract(text):
    if not text:
        return []
    t, found = text.lower(), []
    for skill, rx in COMPILED:
        m = rx.search(t)
        if not m:
            continue
        if skill == "r_lang" and not R_CONTEXT.search(text):
            continue
        found.append(skill)
        t = (t[:m.start()]
             + " " * (m.end() - m.start())
             + t[m.end():])
    return sorted(set(found))


# COMMAND ----------

# MAGIC %md ## Tests 

# COMMAND ----------

def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    return cond

ok = True
ok &= check("sql not matched inside postgresql",
            extract("We use PostgreSQL daily") == ["postgres"])
ok &= check("java not matched inside javascript",
            "java" not in extract("Strong JavaScript skills"))
ok &= check("german posting parsed",
            set(extract("Kenntnisse in Python und Erfahrung "
                        "mit Apache Spark"))
            == {"python", "spark"})
ok &= check("R not matched in R&D",
            "r_lang" not in extract("Our R&D team in Berlin"))
ok &= check("R matched with tech context",
            "r_lang" in extract("Sprachen: Python, R, SQL"))
ok &= check("php+mysql from a real snippet",
            set(extract("Kernsystem (PHP / MySQL) und einen "
                        "modernen Node-Stack"))
            == {"php", "mysql"})

ok &= check("LoRaWAN is not fine-tuning",
            "finetuning" not in extract("LoRaWAN Sensorik im Feld"))
ok &= check("LoRA is fine-tuning",
            "finetuning" in extract("LoRA fine-tuning of open models"))
ok &= check("Claude the first name is not a vendor",
            "openai_api" not in extract("Ansprechpartner: Claude Meier"))
ok &= check("Claude API is a vendor",
            "openai_api" in extract("Erfahrung mit der Claude API"))
ok &= check("rag and llm are now separate skills",
            set(extract("RAG pipelines on top of LLMs"))
            == {"rag", "llm"})
ok &= check("azure ml does not swallow the azure cloud skill",
            extract("Azure ML") == ["azureml"])
ok &= check("genai german spelling",
            "genai" in extract("Erfahrung mit generativer KI ist "
                               "ein Plus")
            or "genai" in extract("Generative KI Projekte"))
ok &= check("mlops stack parsed",
            set(extract("MLflow und Kubeflow auf Kubernetes"))
            == {"mlflow", "kubeflow", "kubernetes"})
ok &= check("vector db recognised",
            "vectordb" in extract("Wir nutzen pgvector fuer Suche"))
ok &= check("transformers library goes to huggingface not deep learning",
            "huggingface" in extract("Hugging Face transformers library"))


# =====================================================
# ENGINEERING SKILL EXTRACTION TESTS
# =====================================================

ok &= check(
    "C++ detected",
    "cpp" in extract("Experience with C++ and Python")
)

ok &= check(
    "ROS2 detected",
    "ros2" in extract("Development with ROS2")
)

ok &= check(
    "ROS 2 detected with space",
    "ros2" in extract("Experience with ROS 2")
)

ok &= check(
    "ROS2 not incorrectly classified as ROS",
    "ros" not in extract("Experience with ROS2")
)

ok &= check(
    "SLAM detected",
    "slam" in extract("SLAM algorithm development")
)

ok &= check(
    "Robotics stack detected",
    {"cpp", "python", "ros2", "slam"}.issubset(
        set(extract("C++, Python, ROS2 and SLAM"))
    )
)

ok &= check(
    "Sensor technologies detected",
    {"lidar", "imu", "radar"}.issubset(
        set(extract("Experience with LiDAR, IMU and Radar"))
    )
)

ok &= check(
    "German sensor terminology detected",
    "sensor_fusion" in extract(
        "Erfahrung in der Sensorfusion"
    )
)

ok &= check(
    "Computer vision tools detected",
    {"cv", "opencv", "object_detection"}.issubset(
        set(extract(
            "Computer Vision, OpenCV and Object Detection"
        ))
    )
)

ok &= check(
    "Visual inertial odometry detected",
    "vio" in extract(
        "Visual-Inertial Odometry for autonomous robots"
    )
)

ok &= check(
    "MATLAB and Simulink detected",
    {"matlab", "simulink"}.issubset(
        set(extract("Modellentwicklung mit MATLAB und Simulink"))
    )
)

ok &= check(
    "PID controller detected",
    "pid" in extract("Design of a PID controller")
)

ok &= check(
    "Kalman filter detected",
    "kalman_filter" in extract("Kalman filter implementation")
)

ok &= check(
    "Extended Kalman Filter detected",
    "ekf" in extract("Extended Kalman Filter development")
)

ok &= check(
    "German control engineering detected",
    "control_systems" in extract(
        "Entwicklung in der Regelungstechnik"
    )
)

ok &= check(
    "FEM simulation tools detected",
    {"fem", "abaqus", "code_aster"}.issubset(
        set(extract("FEM simulations using Abaqus and Code_Aster"))
    )
)

ok &= check(
    "Digital Twin detected",
    "digital_twin" in extract("Digital Twin Development")
)

ok &= check(
    "German Digital Twin detected",
    "digital_twin" in extract("Digitaler Zwilling")
)

ok &= check(
    "Drone software stack detected",
    {"px4", "mavlink", "ros2"}.issubset(
        set(extract("PX4, MAVLink and ROS2 integration"))
    )
)

ok &= check(
    "ADAS detected",
    "adas" in extract("ADAS development")
)

ok &= check(
    "Automotive functional safety detected",
    {"functional_safety", "iso26262"}.issubset(
        set(extract("Functional Safety according to ISO 26262"))
    )
)

ok &= check(
    "German automation skills detected",
    {"plc", "tia_portal"}.issubset(
        set(extract("SPS Programmierung mit TIA Portal"))
    )
)

ok &= check(
    "Embedded systems skills detected",
    {"embedded_systems", "stm32"}.issubset(
        set(extract("Embedded Systems development with STM32"))
    )
)

ok &= check(
    "Linux detected",
    "linux" in extract("Development under Ubuntu")
)

ok &= check(
    "Reinforcement learning detected",
    "reinforcement_learning" in extract(
        "Reinforcement Learning for robot control"
    )
)

ok &= check(
    "Multi-agent pathfinding detected",
    "mapf" in extract("Multi-Agent Pathfinding")
)

ok &= check(
    "Empty description handled correctly",
    extract(None) == []
)

# =====================================================
# FINAL TEST ASSERTION
# =====================================================

assert ok, "matcher tests failed, fix the dictionary first"

print()
print("All original and engineering skill tests passed.")
print(f"Total registered skills: {len(SKILLS)}")
print(f"Total skill categories: {len(set(CATEGORY.values()))}")


# COMMAND ----------

# MAGIC %md ## Apply

# COMMAND ----------

extract_udf = F.udf(extract, ArrayType(StringType()))

p = spark.table(f"{SILVER}.postings")

skills = (p
  .withColumn("text", F.concat_ws(" . ", "title_raw", "description"))
  .withColumn("skill", F.explode(extract_udf(F.col("text"))))
  .withColumn("skill_category",
              F.create_map(*[F.lit(x) for kv in CATEGORY.items()
                             for x in kv])[F.col("skill")])
  .select("posting_id", "skill", "skill_category",
          "role_family", "seniority", "city", "language",
          "is_agency", "age_days", "snapshot_date"))

(skills.write.mode("overwrite")
       .option("overwriteSchema", "true")
       .saveAsTable(f"{SILVER}.posting_skills"))