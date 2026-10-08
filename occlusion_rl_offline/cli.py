"""Single offline CLI for data, teacher and agent workflows."""
import argparse
import os
from .runtime import resolve_device
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .config import load_config, resolve
from .cohort import index_prepared, prepare_batch, read_manifest
from .data import load_case, sample_sdf
from .inference import infer_agent, score_case
from .prepare import prepare_case
from .simulate import perturb_case


def _path(value, fallback):
    return resolve(fallback if value is None else value)


def _snapshot(out, config):
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.snapshot.json').write_text(json.dumps(config, indent=2))


def _run(module, args):
    subprocess.run([sys.executable, '-m', module] + [str(x) for x in args], check=True)


def _teacher_checkpoint(config, override):
    if override:
        return resolve(override)
    stage = 'joint' if config['teacher'].get('trainer') == 'joint' else 'calibration'
    trained = resolve(config['paths']['run_dir']) / ('teacher/%s/best.pt' % stage)
    return trained if trained.exists() else resolve(config['paths']['packaged_teacher'])


def _agent_checkpoint(config, override):
    if override:
        return resolve(override)
    trained = resolve(config['paths']['run_dir']) / 'agent/agent.pt'
    return trained if trained.exists() else resolve(config['paths']['packaged_agent'])


def _agent_args(config, case, teacher, out, eval_only=False, checkpoint=None,
                train_cases=None, eval_cases=None):
    a = config['agent']
    command = ['--case', case, '--teacher', teacher, '--out', out,
               '--points', a['fps_points'], '--teacher-samples', a['fps_points'],
               '--seed', a['seed'],
               '--model-options', json.dumps(config['model']),
               '--policy-width', a['policy_width'],
               '--initial-log-std', a['initial_log_std'],
               '--max-rotation', a['start_rotation_deg'],
               '--max-translation', a['start_translation_mm'],
               '--rotation-step-deg', a['action_rotation_step_deg'],
               '--translation-step-mm', a['action_translation_step_mm'],
               '--horizon', a['horizon'], '--envs', a['parallel_envs'],
               '--iterations', a['iterations'], '--ppo-epochs', a['ppo_epochs'],
               '--minibatch', a['minibatch'], '--lr', a['learning_rate'],
               '--ppo-clip', a['ppo_clip'], '--value-coef', a['value_coef'],
               '--entropy-coef', a['entropy_coef'], '--max-grad-norm', a['max_grad_norm'],
               '--reward-scale', a['reward_scale'], '--gamma', a['gamma'],
               '--gae-lambda', a['gae_lambda'], '--log-every', a['log_every'],
               '--validation-seed-offset', a['validation_seed_offset'],
               '--success-rotation-deg', a['success_rotation_deg'],
               '--success-translation-mm', a['success_translation_mm']]
    if eval_only:
        command += ['--eval-only', '--checkpoint', checkpoint,
                    '--eval-episodes', a['test_episodes'],
                    '--eval-seed-offset', a['test_seed_offset']]
    else:
        command += ['--eval-episodes', a['validation_episodes']]
        if checkpoint:
            command += ['--checkpoint', checkpoint]
    if a.get('score_gated_actions', False):
        command += ['--score-gated-actions']
    if a.get('training_pose_bands'):
        command += ['--training-pose-bands', json.dumps(a['training_pose_bands'])]
    if train_cases is not None:
        command += ['--train-cases', json.dumps([str(x) for x in train_cases])]
    if eval_cases is not None:
        command += ['--eval-cases', json.dumps([str(x) for x in eval_cases])]
    return command


def main():
    parser = argparse.ArgumentParser(prog='occlusion-rl', description=__doc__)
    parser.add_argument('--config', default='configs/default.json',
                        help='JSON config path relative to repository root')
    parser.add_argument('--device', default=None, help='auto, cpu, cuda or cuda:N (before subcommand)')
    subs = parser.add_subparsers(dest='command', required=True)
    prepare = subs.add_parser('prepare', help='Build case.npz from aligned upper/lower STLs')
    prepare.add_argument('--upper', default=None)
    prepare.add_argument('--lower', default=None)
    prepare.add_argument('--out', default=None)
    batch = subs.add_parser('prepare-batch', help='Prepare case_id/upper.stl and lower.stl folders')
    batch.add_argument('--input-dir', required=True)
    batch.add_argument('--out', default=None)
    batch.add_argument('--skip-existing', action='store_true')
    index = subs.add_parser('index-cases', help='Manifest from existing case_id/case.npz folders')
    index.add_argument('--input-dir', required=True)
    index.add_argument('--out', default=None)
    inspect = subs.add_parser('inspect-data', help='Validate a prepared case and print nearest-point distance statistics')
    inspect.add_argument('--case', default=None)
    perturb = subs.add_parser('perturb', help='Make a posed upper case for offline inference testing')
    perturb.add_argument('--case', default=None)
    perturb.add_argument('--upper-stl', default=None)
    perturb.add_argument('--out-case', default=None)
    perturb.add_argument('--rotation-rvec-deg', nargs=3, type=float, required=True)
    perturb.add_argument('--translation-mm', nargs=3, type=float, required=True)
    teacher = subs.add_parser('train-teacher', help='Train ranking and calibrated teacher')
    teacher.add_argument('--case', default=None)
    teacher.add_argument('--manifest', default=None)
    teacher.add_argument('--source', default=None)
    teacher.add_argument('--out', default=None)
    agent = subs.add_parser('train-agent', help='Train PPO with teacher-score delta as sole reward')
    agent.add_argument('--case', default=None)
    agent.add_argument('--manifest', default=None)
    agent.add_argument('--teacher', default=None)
    agent.add_argument('--out', default=None)
    agent.add_argument('--checkpoint', default=None, help='Optional agent checkpoint to initialize from')
    score = subs.add_parser('score', help='Score the current upper pose in a prepared case')
    score.add_argument('--case', default=None)
    score.add_argument('--teacher', default=None)
    evaluate = subs.add_parser('evaluate-agent', help='Evaluate new perturbations of an aligned case')
    evaluate.add_argument('--case', default=None)
    evaluate.add_argument('--manifest', default=None)
    evaluate.add_argument('--teacher', default=None)
    evaluate.add_argument('--agent', default=None)
    evaluate.add_argument('--out', default=None)
    infer = subs.add_parser('infer-agent', help='Recover a posed upper jaw without target pose input')
    infer.add_argument('--case', default=None)
    infer.add_argument('--upper-stl', default=None)
    infer.add_argument('--teacher', default=None)
    infer.add_argument('--agent', default=None)
    infer.add_argument('--out', default=None)
    infer.add_argument('--steps', type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    device = resolve_device(config,args.device)
    config.setdefault('runtime',{})['device']=device
    os.environ['OCCLUSION_DEVICE']=device
    case = _path(getattr(args, 'case', None), config['paths']['case'])
    run_dir = resolve(config['paths']['run_dir'])
    if getattr(args, 'manifest', None) and getattr(args, 'case', None):
        parser.error('Use either --case or --manifest, not both')
    cohort = read_manifest(resolve(args.manifest)) if getattr(args, 'manifest', None) else None
    if args.command == 'prepare':
        upper = _path(args.upper, config['paths']['upper_stl'])
        lower = _path(args.lower, config['paths']['lower_stl'])
        out = _path(args.out, run_dir / 'prepared/case.npz')
        _, audit = prepare_case(upper, lower, out, config['prepare'])
        print(json.dumps({'case': str(out), 'audit': audit}, indent=2))
    elif args.command == 'prepare-batch':
        c = config['cohort']
        out = _path(args.out, run_dir / 'cohort')
        manifest, counts = prepare_batch(resolve(args.input_dir), out, config['prepare'],
                                         c['upper_filename'], c['lower_filename'],
                                         c['split_seed'], c['validation_fraction'],
                                         c['test_fraction'], args.skip_existing)
        print(json.dumps({'manifest': str(manifest), 'counts': counts}, indent=2))
    elif args.command == 'index-cases':
        c = config['cohort']
        out = _path(args.out, run_dir / 'cohort/manifest.json')
        manifest, counts = index_prepared(resolve(args.input_dir), out,
                                          c['split_seed'], c['validation_fraction'],
                                          c['test_fraction'])
        print(json.dumps({'manifest': str(manifest), 'counts': counts}, indent=2))
    elif args.command == 'inspect-data':
        data = load_case(case)
        probe = sample_sdf(data, data['xyz'][:min(4096, len(data['xyz']))])
        print(json.dumps({'case': str(case), 'upper_points': len(data['xyz']),
                          'lower_points': len(data['lower_xyz']),
                          'distance_method': 'nearest lower mesh vertex; vertex-normal sign',
                          'probe_negative_fraction': float((probe < 0).mean()),
                          'probe_within_0_6mm': float((np.abs(probe) < .6).mean())}, indent=2))
    elif args.command == 'perturb':
        out_case = _path(args.out_case, run_dir / 'perturbed/case.npz')
        upper_stl = _path(args.upper_stl, config['paths']['upper_stl'])
        result = perturb_case(case, out_case, args.rotation_rvec_deg,
                              args.translation_mm, upper_stl)
        print(json.dumps(result, indent=2))
    elif args.command == 'train-teacher':
        t = config['teacher']
        out = _path(args.out, run_dir / 'teacher')
        _snapshot(out, config)
        if t.get('trainer') == 'joint':
            if not args.manifest:
                parser.error('Joint teacher training requires --manifest')
            source = _path(args.source, config['paths'].get('optimization_source_teacher',config['paths']['packaged_teacher']))
            _run('occlusion_rl_offline.joint_teacher',
                 ['--config', args.config, '--manifest', args.manifest, '--source', source,
                  '--out', out / 'joint', '--mode', t.get('joint_mode','consistent')])
            return
        rank = out / 'ranking'
        calibrated = out / 'calibration'
        source = _path(args.source, config['paths']['initial_teacher'])
        model_options = json.dumps(config['model'])
        split_args = (['--train-cases', json.dumps([str(x) for x in cohort['train']]),
                       '--val-cases', json.dumps([str(x) for x in cohort['val']])]
                      if cohort else [])
        _run('occlusion_rl_offline.train_case_teacher',
             ['--case', case, '--source', source, '--out', rank,
              '--epochs', t['ranking_epochs'], '--repeats', t['ranking_pairs_per_epoch'],
              '--validation-repeats', t.get('ranking_validation_pairs_per_case', 128),
              '--points', t['fps_points'], '--samples', t['fps_points'],
              '--batch', t['ranking_batch'], '--lr', t['ranking_lr'], '--seed', t['seed'],
              '--weight-decay', t['ranking_weight_decay'],
              '--near-translation-mm', t['ranking_near_translation_mm'],
              '--near-rotation-deg', t['ranking_near_rotation_deg'],
              '--far-magnitude-min', t['ranking_far_magnitude_min'],
              '--far-magnitude-max', t['ranking_far_magnitude_max'],
              '--ranking-pose-bands', json.dumps(t.get('ranking_pose_bands')),
              '--margin', t['ranking_margin'],
              '--anchor-weight', t['ranking_anchor_weight'],
              '--model-options', model_options] + split_args +
             (['--from-scratch'] if args.source is None else []))
        _run('occlusion_rl_offline.calibrate_teacher',
             ['--case', case, '--source', rank / 'best.pt', '--out', calibrated,
              '--epochs', t['calibration_epochs'],
              '--train-count', t['calibration_poses_per_epoch'],
              '--val-count', t['calibration_validation_poses'],
              '--points', t['fps_points'], '--samples', t['fps_points'],
              '--batch', t['calibration_batch'], '--lr', t['calibration_lr'],
              '--weight-decay', t['calibration_weight_decay'],
              '--smooth-l1-beta', t['calibration_smooth_l1_beta'],
              '--max-grad-norm', t['calibration_max_grad_norm'],
              '--pose-bands', json.dumps(t['calibration_pose_bands']),
              '--quality-rotation-scale-deg', t['quality_rotation_scale_deg'],
              '--quality-translation-scale-mm', t['quality_translation_scale_mm'],
              '--seed', t['seed'], '--model-options', model_options] + split_args)
        print(json.dumps({'ranking_checkpoint': str(rank / 'best.pt'),
                          'teacher_checkpoint': str(calibrated / 'best.pt')}))
    elif args.command == 'train-agent':
        out = _path(args.out, run_dir / 'agent')
        _snapshot(out, config)
        teacher_path = _teacher_checkpoint(config, args.teacher)
        if config['agent'].get('trainer') == 'continuous':
            if not args.manifest:
                parser.error('Continuous PPO requires --manifest for separate train/validation cases')
            command = ['--config', args.config, '--manifest', args.manifest,
                       '--teacher', teacher_path, '--out', out]
            if args.checkpoint:
                command += ['--resume', resolve(args.checkpoint)]
            _run('occlusion_rl_offline.optimized_ppo', command)
            return
        command = _agent_args(config, case, teacher_path, out,
                              checkpoint=resolve(args.checkpoint) if args.checkpoint else None,
                              train_cases=cohort['train'] if cohort else None,
                              eval_cases=cohort['val'] if cohort else None)
        _run('occlusion_rl_offline.rl', command)
        print(json.dumps({'agent_checkpoint': str(out / 'agent.pt'),
                          'teacher_checkpoint': str(teacher_path)}))
    elif args.command == 'score':
        teacher_path = _teacher_checkpoint(config, args.teacher)
        print(json.dumps(score_case(case, teacher_path, config), indent=2))
    elif args.command == 'evaluate-agent':
        out = _path(args.out, run_dir / 'agent')
        teacher_path = _teacher_checkpoint(config, args.teacher)
        agent_path = _agent_checkpoint(config, args.agent)
        if config['agent'].get('trainer') == 'continuous':
            if not args.manifest:
                parser.error('Continuous evaluation requires --manifest')
            _run('occlusion_rl_offline.evaluation',
                 ['--config', args.config, '--manifest', args.manifest, '--teacher', teacher_path,
                  '--agent', agent_path, '--split', 'test', '--episodes', config['agent']['test_episodes'],
                  '--horizons', config['agent']['horizon'], '--out', out / 'test_evaluation.json'])
            return
        _run('occlusion_rl_offline.rl',
             _agent_args(config, case, teacher_path, out, eval_only=True,
                         checkpoint=agent_path, eval_cases=cohort['test'] if cohort else None))
    elif args.command == 'infer-agent':
        teacher_path = _teacher_checkpoint(config, args.teacher)
        agent_path = _agent_checkpoint(config, args.agent)
        out = _path(args.out, config['inference']['out_dir'])
        upper = resolve(args.upper_stl) if args.upper_stl else None
        result = infer_agent(case, teacher_path, agent_path, out, config, upper, args.steps)
        print(json.dumps({'trajectory': str(out / 'trajectory.json'),
                          'upper_recovered_stl': result.get('upper_recovered_stl'),
                          'initial_teacher_logit': result['initial_teacher_logit'],
                          'final_teacher_logit': result['final_teacher_logit']}, indent=2))


if __name__ == '__main__':
    main()
