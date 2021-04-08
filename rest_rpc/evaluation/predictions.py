#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import os
import random
from logging import NOTSET

# Libs
from flask import request
from flask_restx import Namespace, Resource, fields

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import TopicalPayload
from rest_rpc.training.core.server import align_proc
from rest_rpc.training.core.utils import RPCFormatter
from rest_rpc.evaluation.core.server import evaluate_proc
from rest_rpc.evaluation.validations import meta_stats_model
from synarchive.connection import (
    ProjectRecords,
    ExperimentRecords,
    RunRecords,
    ParticipantRecords,
    RegistrationRecords,
    TagRecords
)
from synarchive.training import AlignmentRecords, ModelRecords
from synarchive.evaluation import PredictionRecords

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

ns_api = Namespace(
    "predictions", 
    description='API to faciliate model inference in a REST-RPC Grid.'
)

grid_idx = app.config['GRID']

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

logging = app.config['NODE_LOGGER'].synlog
logging.debug("evaluation/predictions.py logged", Description="No Changes")

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

payload_formatter = TopicalPayload(
    subject=prediction_records.subject, 
    namespace=ns_api, 
    model=pred_output_model
)

#############
# Resources #
#############

@ns_api.route('/', defaults={'project_id': None, 'expt_id': None, 'run_id': None})
@ns_api.route('/<project_id>', defaults={'expt_id': None, 'run_id': None})
@ns_api.route('/<project_id>/<expt_id>', defaults={'run_id': None})
@ns_api.route('/<project_id>/<expt_id>/<run_id>')
@ns_api.response(403, 'Feature drift detected')
@ns_api.response(404, 'Predictions not found')
@ns_api.response(500, 'Internal failure')
class Predictions(Resource):
    """ 
    Handles model inference within the PySyft grid. Model inference is done
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
    def get(self, participant_id, collab_id, project_id, expt_id, run_id):
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

            logging.info(
                "Participant '{}' >|< Collaboration '{}' > Project '{}' > Experiment '{}' > Run '{}' >|< Predictions: Bulk record retrieval successful!".format(
                    participant_id, collab_id, project_id, expt_id, run_id
                ),
                code=200, 
                description="Predictions for participant '{}' under collaboration '{}''s project '{}' using experiment '{}' and run '{}' was successfully retrieved!".format(
                    participant_id, collab_id, project_id, expt_id, run_id
                ),
                ID_path=SOURCE_FILE,
                ID_class=Predictions.__name__, 
                ID_function=Predictions.get.__name__,
                **request.view_args
            )
            
            return success_payload, 200

        else:
            logging.error(
                "Participant '{}' >|< Collaboration '{}' > Project '{}' > Experiment '{}' > Run '{}' >|< Predictions: Bulk record retrieval failed!".format(
                    participant_id, collab_id, project_id, expt_id, run_id
                ),
                code=404, 
                description=f"Predictions do not exist for specified keyword filters!", 
                ID_path=SOURCE_FILE,
                ID_class=Predictions.__name__, 
                ID_function=Predictions.get.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=404, 
                message=f"Predictions do not exist for specified keyword filters!"
            )


    @ns_api.doc("trigger_predictions")
    @ns_api.expect(pred_input_model)
    @ns_api.marshal_with(payload_formatter.plural_model)
    def post(self, participant_id, collab_id, project_id, expt_id, run_id):
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
        # Populate grid-initialising parameters
        init_params = request.json

        auto_align = init_params['auto_align']
        is_dockerised = init_params['dockerised']
        new_pred_tags = init_params['tags']

        logging.debug(
            "Participant '{}' >|< Collaboration '{}' > Project '{}' > Experiment '{}' > Run '{}' >|< Predictions: Input keys tracked.".format(
                participant_id, collab_id, project_id, expt_id, run_id
            ),
            keys=request.view_args, 
            ID_path=SOURCE_FILE,
            ID_class=Predictions.__name__, 
            ID_function=Predictions.post.__name__,
            **request.view_args
        )

        # Update prediction tags for all queried projects
        for queried_project_id, tags in new_pred_tags.items():
            tag_records.update(
                collab_id=collab_id,
                project_id=queried_project_id, 
                participant_id=participant_id,
                updates={'predict': tags}
            )

        # Participant's POV: Retrieve all projects within collaborations
        # registered under participant. If specific project was declared, 
        # collapse inference space
        key_filter = {
            'participant_id': participant_id,
            'collab_id': collab_id,
            'project_id': project_id
        }
        key_filter = {k:v for k,v in key_filter.items() if v is not None}

        logging.debug(
            "Participant '{}' >|< Collaboration '{}' > Project '{}' > Experiment '{}' > Run '{}' >|< Predictions: key_filter tracked.".format(
                participant_id, collab_id, project_id, expt_id, run_id
            ),
            key_filter=key_filter, 
            ID_path=SOURCE_FILE,
            ID_class=Predictions.__name__, 
            ID_function=Predictions.post.__name__,
            **request.view_args
        )

        participant_registrations = registration_records.read_all(
            filter=key_filter
        )

        logging.debug(
            "Participant '{}' >|< Collaboration '{}' > Project '{}' > Experiment '{}' > Run '{}' >|< Predictions: Participant registrations tracked.".format(
                participant_id, collab_id, project_id, expt_id, run_id
            ),
            registrations=participant_registrations, 
            ID_path=SOURCE_FILE,
            ID_class=Predictions.__name__, 
            ID_function=Predictions.post.__name__,
            **request.view_args
        )

        project_combinations = {}
        for registration in participant_registrations:

            registered_project_id = registration['project']['key']['project_id']

            # Retrieves expt-run supersets (i.e. before filtering for relevancy)
            retrieved_project = project_records.read(
                collab_id=collab_id,
                project_id=registered_project_id
            )
            project_action = retrieved_project['action']
            experiments = retrieved_project['relations']['Experiment']
            runs = retrieved_project['relations']['Run']

            # If specific experiment was declared, collapse inference space
            if expt_id:

                retrieved_expt = expt_records.read(
                    collab_id=collab_id,
                    project_id=project_id, 
                    expt_id=expt_id
                )
                runs = retrieved_expt['relations']['Run']
                experiments = [retrieved_expt]

                # If specific run was declared, further collapse inference space
                if run_id:

                    retrieved_run = run_records.read(
                        collab_id=collab_id,
                        project_id=project_id, 
                        expt_id=expt_id,
                        run_id=run_id
                    )
                    runs = [retrieved_run]

            # Retrieve all participants' metadata enrolled for curr project
            project_registrations = registration_records.read_all(
                filter={
                    'collab_id': collab_id,
                    'project_id': registered_project_id
                }
            )
            unaligned_grids = rpc_formatter.extract_grids(project_registrations)
            selected_unaligned_grid = unaligned_grids[grid_idx]

            ###########################
            # Implementation Footnote #
            ###########################

            # [Cause]
            # Decoupling of MFA from inference should be made more explicit.

            # [Problems]
            # Auto-alignment is not scalable to datasets that have too many 
            # features and can consume too much computation resources such that 
            # the container will crash.

            # [Solution]
            # Explicitly declare new state parameters that allows sections of
            # alignment procedure to be skipped when necessary, provided that 
            # the declared model parameters are CORRECT!!! If `auto_align` is 
            # true, then MFA will be performed to obtain alignment indexes for 
            # the newly declared prediction data tags in preparation for 
            # inference. If `auto-align` is false, then MFA is skipped 
            # (i.e. inference data is ASSUMED to have the same structure as 
            # that of training & validation data). However, the model should 
            # not be allowed to mutate, and so `auto_fix` must be de-activated. 

            spacer_collection, _, _ = align_proc(
                grid=selected_unaligned_grid, 
                kwargs={
                    'experiments': experiments,
                    'auto_align': auto_align, # MFA toggled dependent on participant
                    'auto_fix': False         # model cannot be allowed to mutate
                }
            )

            for p_id, spacer_idxs in spacer_collection.items():

                if p_id != participant_id:
                    
                    ###########################
                    # Implementation Footnote #
                    ###########################

                    # [Cause]
                    # Alignments are dynamically generated based on the
                    # retrieved state of headers from the workers. When
                    # declaring new prediction tags, the current state of
                    # alignments archived have to be re-evaluated for changes
                    # to accomodate the new dataset(s).

                    # [Problems]
                    # Auto re-alignment should not affect existing datasets.
                    # This is because the global model has been calibrated with
                    # their feature set, and any modification to the feature
                    # set will render the trained model unusable, as the
                    # input/output structures of the model architecture will
                    # become mis-aligned.

                    # [Solution]
                    # Do NOT allow feature drift. If participant declares a
                    # prediction tag that contains feature/label classes that
                    # were unaccounted for in the training of the current 
                    # model, then reject the inference request.

                    retrieved_alignments = alignment_records.read(
                        collab_id=collab_id,
                        project_id=registered_project_id,
                        participant_id=p_id
                    )
                    retrieved_spacer_idxs = rpc_formatter.strip_keys(
                        record=retrieved_alignments,
                        concise=True
                    )

                    if retrieved_spacer_idxs != spacer_idxs:
                        logging.error(
                            "Participant '{}' >|< Collaboration '{}' > Project '{}' > Experiment '{}' > Run '{}' >|< Predictions: Feature drift detected!".format(
                                participant_id, collab_id, project_id, expt_id, run_id
                            ),
                            description="Dataset characteristics are incompatible with current model.",
                            spacer_idxs=spacer_idxs,
                            ID_path=SOURCE_FILE,
                            ID_class=Predictions.__name__, 
                            ID_function=Predictions.post.__name__,
                            **request.view_args
                        )
                        ns_api.abort(
                            code=403,
                            message="Feature drift detected! Dataset characteristics are incompatible with current model."
                        )

                else:
                    logging.debug(
                        "Participant '{}' >|< Collaboration '{}' > Project '{}' > Experiment '{}' > Run '{}' >|< Predictions: Spacer Indexes tracked.".format(
                            participant_id, collab_id, project_id, expt_id, run_id
                        ),
                        spacer_idxs=spacer_idxs,
                        ID_path=SOURCE_FILE,
                        ID_class=Predictions.__name__, 
                        ID_function=Predictions.post.__name__,
                        **request.view_args
                    )

                    alignment_records.update(
                        collab_id=collab_id,
                        project_id=registered_project_id,
                        participant_id=p_id,
                        updates=spacer_idxs
                    ) 

            # Retrieve registrations updated with possibly new alignments
            updated_registrations = registration_records.read_all(
                filter={
                    'collab_id': collab_id,
                    'project_id': registered_project_id
                }
            )
            aligned_grids = rpc_formatter.extract_grids(updated_registrations)
            selected_grid = aligned_grids[grid_idx]

            logging.debug(
                "Participant '{}' >|< Collaboration '{}' > Project '{}' > Experiment '{}' > Run '{}' >|< Predictions: Updated project registrations tracked.".format(
                    participant_id, collab_id, project_id, expt_id, run_id
                ),
                updated_registrations=updated_registrations, 
                ID_path=SOURCE_FILE,
                ID_class=Predictions.__name__, 
                ID_function=Predictions.post.__name__,
                **request.view_args
            )

            # Template for initialising FL grid
            kwargs = {
                'action': project_action,
                'auto_align': auto_align,
                'dockerised': is_dockerised,
                'experiments': experiments,
                'runs': runs,
                'participants': [participant_id],
                'metas': ['predict'],
                'version': None # defaults to final state of federated grid
            }
            project_combinations[registered_project_id] = kwargs

        logging.debug(
            "Participant '{}' >|< Collaboration '{}' > Project '{}' > Experiment '{}' > Run '{}' >|< Predictions: Project combinations tracked.".format(
                participant_id, collab_id, project_id, expt_id, run_id
            ),
            project_combinations=project_combinations, 
            ID_path=SOURCE_FILE,
            ID_class=Predictions.__name__, 
            ID_function=Predictions.post.__name__,
            **request.view_args
        )

        completed_inferences = evaluate_proc(
            grid=selected_grid, 
            multi_kwargs=project_combinations
        )

        logging.log(
            level=NOTSET,
            event="Participant '{}' >|< Collaboration '{}' > Project '{}' > Experiment '{}' > Run '{}' >|< Predictions: Completed Inferences tracked.".format(
                participant_id, collab_id, project_id, expt_id, run_id
            ),
            completed_inferences=completed_inferences,
            ID_path=SOURCE_FILE,
            ID_class=Predictions.__name__, 
            ID_function=Predictions.post.__name__,
            **request.view_args
        )
        
        # Store output metadata into database
        retrieved_predictions = []
        for combination_key, inference_stats in completed_inferences.items():

            logging.debug(
                "Participant '{}' >|< Collaboration '{}' > Project '{}' > Experiment '{}' > Run '{}' >|< Predictions: Inference stats tracked.".format(
                    participant_id, collab_id, project_id, expt_id, run_id
                ),
                inference_stats=inference_stats, 
                ID_path=SOURCE_FILE,
                ID_class=Predictions.__name__, 
                ID_function=Predictions.post.__name__,
                **request.view_args
            )

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

        logging.info(
            "Participant '{}' >|< Collaboration '{}' > Project '{}' > Experiment '{}' > Run '{}' >|< Predictions: Record creation successful!".format(
                participant_id, collab_id, project_id, expt_id, run_id
            ),
            description=f"Predictions for participant '{participant_id}' under collaboration '{collab_id}' for project '{project_id}' using experiment '{expt_id}' and run '{run_id}' was successfully collected!",
            code=201, 
            ID_path=SOURCE_FILE,
            ID_class=Predictions.__name__, 
            ID_function=Predictions.post.__name__,
            **request.view_args
        )

        return success_payload, 200
