#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging

# Libs
import jsonschema
import mlflow
from flask import request
from flask_restx import Namespace, Resource, fields

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import (
    TopicalPayload,
    ProjectRecords,
    ExperimentRecords,
    RunRecords,
    ParticipantRecords,
    RegistrationRecords,
    TagRecords
)
from rest_rpc.training.core.feature_alignment import MultipleFeatureAligner
from rest_rpc.training.core.utils import (
    AlignmentRecords, 
    ModelRecords,
    Poller,
    RPCFormatter
)
from rest_rpc.evaluation.core.server import start_proc
from rest_rpc.evaluation.core.utils import PredictionRecords
from rest_rpc.evaluation.validations import meta_stats_model

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns_api = Namespace(
    "predictions", 
    description='API to faciliate model inference in a REST-RPC Grid.'
)

SUBJECT = "Prediction"

db_path = app.config['DB_PATH']
project_records = ProjectRecords(db_path=db_path)
expt_records = ExperimentRecords(db_path=db_path)
run_records = RunRecords(db_path=db_path)
participant_records = ParticipantRecords(db_path=db_path)
registration_records = RegistrationRecords(db_path=db_path)
tag_records = TagRecords(db_path=db_path)
alignment_records = AlignmentRecords(db_path=db_path)
model_records = ModelRecords(db_path=db_path)
prediction_records = PredictionRecords(db_path=db_path)

rpc_formatter = RPCFormatter()

################################################################
# Predictions - Used for marshalling (i.e. moulding responses) #
################################################################

# Marshalling inputs
prediction_tag_field = fields.Wildcard(fields.List(fields.List(fields.String())))
prediction_tag_model = ns_api.model(
    name="prediction_tags",
    model={"*": prediction_tag_field}
)

pred_input_model = ns_api.model(
    name="prediction_input",
    model={
        'auto_align': fields.Boolean(default=True, required=True),
        'dockerised': fields.Boolean(default=False, required=True),
        'tags': fields.Nested(model=prediction_tag_model, skip_none=True)
    }
)

# Marshalling Outputs
# - same `meta_stats_model` retrieved from Validations resource

pred_inferences_model = ns_api.model(
    name="prediction_inferences",
    model={
        'predict': fields.Nested(meta_stats_model, required=True)
    }
)

pred_output_model = ns_api.inherit(
    "prediction_output",
    pred_inferences_model,
    {
        'doc_id': fields.String(),
        'kind': fields.String(),
        'key': fields.Nested(
            ns_api.model(
                name='key',
                model={
                    'participant_id': fields.String(),
                    'project_id': fields.String(),
                    'expt_id': fields.String(),
                    'run_id': fields.String()
                }
            ),
            required=True
        )
    }
)

payload_formatter = TopicalPayload(SUBJECT, ns_api, pred_output_model)

#############
# Resources #
#############

@ns_api.route('/', defaults={'project_id': None, 'expt_id': None, 'run_id': None})
@ns_api.route('/<project_id>', defaults={'expt_id': None, 'run_id': None})
@ns_api.route('/<project_id>/<expt_id>', defaults={'run_id': None})
@ns_api.route('/<project_id>/<expt_id>/<run_id>')
@ns_api.response(404, 'Predictions not found')
@ns_api.response(500, 'Internal failure')
class Predictions(Resource):
    """ Handles model inference within the PySyft grid. Model inference is done
        a series of 8 stages:
        1) User provides prediction tags for specified projects
        2) Specified model architectures are re-loaded alongside its run configs
        3) Federated grid is re-established
        4) Inference is performed on participant's test datasets
        5) `worker/predict` route is activated to export results to worker node
        6) Statistics are computed on the remote node
        7) Statistics are returned as response to the TTP and archived
        8) Statistics are finally retrievable by participant
    """
    
    @ns_api.doc("get_predictions")
    @ns_api.marshal_with(payload_formatter.plural_model)
    def get(self, participant_id, project_id, expt_id, run_id):
        """ Retrieves global model corresponding to experiment and run 
            parameters for a specified project
        """
        filter = {k:v for k,v in request.view_args.items() if v is not None}

        retrieved_predictions = prediction_records.read_all(filter=filter)
        
        if retrieved_predictions:
            
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="predictions.get",
                params=request.view_args,
                data=retrieved_predictions
            )
            return success_payload, 200

        else:
            ns_api.abort(
                code=404, 
                message=f"Predictions do not exist for specified keyword filters!"
            )


    @ns_api.doc("trigger_predictions")
    @ns_api.expect(pred_input_model)
    @ns_api.marshal_with(payload_formatter.plural_model)
    def post(self, participant_id, project_id, expt_id, run_id):
        """ Triggers FL inference for specified project-experiment-run
            combination within a PySyft FL grid. 
            Note: Participants have the option to specify additional datasets
                  here for prediction. However all prediction sets will be
                  collated into a single combined prediction set to facilitate
                  prediction. Also, all prediction tags submitted here will
                  override any previously submitted prediction tags registered
                  for a specified project. This is to prevent accumulation of 
                  unsynced dataset tags (w.r.t participant's node). Hence, if
                  a participant wants to expand their testing datasets by 
                  declaring new tags, the onus is on them to declare all of it
                  per submission.

            Sample payload:

            {
                "auto_align": true,
                "dockerised": true,
                "tags": {
                    "test_project_1": [["non_iid_2"]],
                    "test_project_2": [["non_iid_1"], ["non_iid_2"]]
                }
            }
        """
        auto_align = request.json['auto_align']
        is_dockerised = request.json['dockerised']
        new_pred_tags = request.json['tags']
        logging.debug(f"Keys: {request.view_args}")

        # Update prediction tags for all queried projects
        for queried_project_id, tags in new_pred_tags.items():
            tag_records.update(
                project_id=queried_project_id, 
                participant_id=participant_id,
                updates={'predict': tags}
            )

        # Participant's POV: Retrieve all projects registered under participant
        # If specific project was declared, collapse inference space
        key_filter = {
            'participant_id': participant_id,
            'project_id': project_id
        }
        key_filter = {k:v for k,v in key_filter.items() if v is not None}
        logging.debug(f"{key_filter}")
        participant_registrations = registration_records.read_all(
            filter=key_filter
        )
        logging.debug(participant_registrations)

        project_combinations = {}
        for registration in participant_registrations:

            registered_project_id = registration['project']['key']['project_id']

            # Retrieves expt-run supersets (i.e. before filtering for relevancy)
            retrieved_project = project_records.read(
                project_id=registered_project_id
            )
            project_action = retrieved_project['action']
            experiments = retrieved_project['relations']['Experiment']
            runs = retrieved_project['relations']['Run']

            # If specific experiment was declared, collapse inference space
            if expt_id:

                retrieved_expt = expt_records.read(
                    project_id=project_id, 
                    expt_id=expt_id
                )
                runs = retrieved_expt.pop('relations')['Run']
                experiments = [retrieved_expt]

                # If specific run was declared, further collapse inference space
                if run_id:

                    retrieved_run = run_records.read(
                        project_id=project_id, 
                        expt_id=expt_id,
                        run_id=run_id
                    )
                    retrieved_run.pop('relations')
                    runs = [retrieved_run]

            # Retrieve all participants' metadata enrolled for curr project
            project_registrations = registration_records.read_all(
                filter={'project_id': registered_project_id}
            )
            
            poller = Poller(project_id=registered_project_id)
            all_metadata = poller.poll(project_registrations)

            logging.debug(f"All metadata polled: {all_metadata}")

            (X_data_headers, y_data_headers,
             key_sequences, _) = rpc_formatter.aggregate_metadata(all_metadata)

            ###########################
            # Implementation Footnote #
            ###########################

            # [Cause]
            # Decoupling of MFA from inference should be made more explicit

            # [Problems]
            # Auto-alignment is not scalable to datasets that have too many 
            # features and can consume too much computation resources such that 
            # the container will crash.

            # [Solution]
            # Explicitly declare a new state parameter that allows the alignment
            # procedure to be skipped when necessary, provided that the declared
            # model parameters are CORRECT!!! If `auto-align` is true, then MFA 
            # will be performed to obtain alignment indexes for the newly 
            # declared prediction data tags in preparation for inference. If 
            # `auto-align` is false, then MFA is skipped (i.e. inference data is
            # ASSUMED to have the same structure as that of training & 
            # validation data)

            if auto_align:
                X_mfa_aligner = MultipleFeatureAligner(headers=X_data_headers)
                X_mf_alignments = X_mfa_aligner.align()

                y_mfa_aligner = MultipleFeatureAligner(headers=y_data_headers)
                y_mf_alignments = y_mfa_aligner.align()

                spacer_collection = rpc_formatter.alignment_to_spacer_idxs(
                    X_mf_alignments=X_mf_alignments,
                    y_mf_alignments=y_mf_alignments,
                    key_sequences=key_sequences
                )

                for p_id, spacer_idxs in spacer_collection.items():

                    logging.debug(f"Spacer Indexes: {spacer_idxs}")
                    alignment_records.update(
                        project_id=registered_project_id,
                        participant_id=p_id,
                        updates=spacer_idxs
                    ) 

                logging.debug(f"Alignments: {alignment_records.read_all(filter={'project_id': registered_project_id, 'participant_id': participant_id})}")

            updated_project_registrations = registration_records.read_all(
                filter={'project_id': registered_project_id}
            )

            logging.debug(f"Project registrations: {updated_project_registrations}")
            # Template for initialising FL grid
            kwargs = {
                'action': project_action,
                'auto_align': auto_align,
                'dockerised': is_dockerised,
                'experiments': experiments,
                'runs': runs,
                'registrations': updated_project_registrations,
                'participants': [participant_id],
                'metas': ['predict'],
                'version': None # defaults to final state of federated grid
            }
            project_combinations[registered_project_id] = kwargs

        logging.debug(f"{project_combinations}")

        completed_inferences = start_proc(project_combinations)
        logging.debug(f"Completed Inferences: {completed_inferences}")
        
        # Store output metadata into database
        retrieved_predictions = []
        for combination_key, inference_stats in completed_inferences.items():
            logging.debug(f"Inference stats: {inference_stats}")

            combination_key = (participant_id,) + combination_key

            new_prediction = prediction_records.create(
                *combination_key,
                details=inference_stats[participant_id]
            )

            retrieved_prediction = prediction_records.read(*combination_key)

            assert new_prediction.doc_id == retrieved_prediction.doc_id
            retrieved_predictions.append(retrieved_prediction)

        success_payload = payload_formatter.construct_success_payload(
            status=200,
            method="predictions.post",
            params=request.view_args,
            data=retrieved_predictions
        )
        return success_payload, 200
