# Jetson Orin Nano Super Deployment Checklist

## Pre-Deployment (Development PC)

- [x] Database schema includes `source` column for mode tracking
- [x] All 3 detection modes (SELECT, ACTIVITY, AUTO) pass individually
- [x] Results API endpoint (`/api/results`) returns filtered results correctly
- [x] Frontend fetch calls include `credentials: 'include'` for session auth
- [x] Mixed precision (FP16) inference code added to app.py
- [x] Jetson optimization module (jetson_optimization.py) created
- [x] TorchScript compilation path implemented
- [x] Benchmark script (benchmark_jetson.py) shows baseline metrics
- [x] All Python files pass syntax validation
- [x] Environment variable `JETSON_OPTIMIZED` added to control optimizations

## Hardware Setup (Jetson Orin Nano Super)

- [ ] Jetson Orin Nano Super flashed with JetPack 5.1.1+ (includes CUDA 12.x)
- [ ] Adequate cooling solution installed (heatsink + fan)
- [ ] Power supply: 5V 4A minimum
- [ ] Network connectivity verified (Ethernet or WiFi)
- [ ] SSH access configured for remote deployment

## Jetson Environment Setup

- [ ] Python 3.8+ verified: `python --version`
- [ ] PyTorch installed: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- [ ] Dependencies installed: `pip install flask flask-cors flask-sqlalchemy python-dotenv`
- [ ] PostgreSQL client configured (or SQLite as fallback)
- [ ] Run model file transferred: `scp run35.pth jetson@<ip>:/path/to/fsl/`

## Code Deployment

- [ ] app.py copied to Jetson
- [ ] model.py copied to Jetson  
- [ ] jetson_optimization.py copied to Jetson
- [ ] All template files (*.html) copied to Jetson
- [ ] Database initialized or migrated from source
- [ ] Requirements.txt installed: `pip install -r requirements.txt`

## Jetson-Specific Configuration

- [ ] Environment variable set: `export JETSON_OPTIMIZED=true`
- [ ] GPU memory fraction configured (0.5 for thermal headroom)
- [ ] Dynamic frequency scaling enabled: `/usr/bin/jetson_clocks`
- [ ] Network connectivity to database server established

## Testing Phase

- [ ] Backend starts without errors: `python app.py`
- [ ] Health check endpoint available: `curl http://localhost:5000/health`
- [ ] Model loads and compiles successfully
- [ ] Mixed precision context activates: Check logs for "AMP enabled"
- [ ] Single inference test: `curl -X POST http://localhost:5000/predict ...`
- [ ] Multiple inference test (stress test 100+ requests)
- [ ] Benchmark runs: `python benchmark_jetson.py`

## Performance Validation

- [ ] FP16 inference time < 25ms per sequence (target: 10-20ms)
- [ ] Throughput > 40 req/sec (target: 50-100 req/sec with optimization)
- [ ] GPU memory usage < 500MB total
- [ ] GPU temperature stays < 60°C under load
- [ ] No thermal throttling detected during sustained inference

## Database Integration

- [ ] Results table connection verified with sample query
- [ ] User authentication working with session tokens
- [ ] Results saved correctly to `source` column (select/activity/auto)
- [ ] API endpoint `/api/results` filters by user_id correctly
- [ ] Historical data accessible from all prediction modes

## Frontend Verification

- [ ] Web interface loads without JavaScript errors
- [ ] Login/authentication works
- [ ] SELECT mode tests record to results.html
- [ ] ACTIVITY mode tests record to results.html
- [ ] AUTO mode tests record to results.html
- [ ] Results page displays all 3 mode tabs (SELECT, ACTIVITY, AUTO)
- [ ] Statistics calculated correctly per mode
- [ ] Session persistence across page reloads

## Performance Monitoring

- [ ] GPU monitoring setup: `nvidia-smi` showing real-time stats
- [ ] Inference latency logging enabled in app.py
- [ ] Database query performance acceptable
- [ ] No memory leaks after 1 hour of continuous operation
- [ ] Docker container option (optional for cleaner deployment)

## Production Hardening

- [ ] HTTPS enabled with self-signed or trusted certificate
- [ ] CORS properly configured for frontend domain
- [ ] Database connection pooling optimized
- [ ] Error handling and logging comprehensive
- [ ] Rate limiting implemented to prevent overload
- [ ] Health check endpoint for monitoring uptime

## Rollback Plan

- [ ] Incremental deployment: Deploy to test instance first
- [ ] Keep previous production state accessible
- [ ] Database backup before switching code
- [ ] Version control commit hash recorded for rollback reference
- [ ] Clear documentation of any breaking changes

## Performance Optimization Results

### Expected Improvements vs Dev PC (Ryzen 5600G):
- Dev PC FP32: 1.92 ms baseline
- Jetson FP16 expected: 10-20 ms (varies by load, optimization level)
- Combined with batch processing: Higher throughput for multiple simultaneous users

### Actual Jetson Results (Post-Deployment):
- FP32 inference time: ___________ ms
- FP16 inference time: ___________ ms  
- TorchScript with FP16: ___________ ms
- Throughput (req/sec): ___________
- GPU memory utilization: ___________%
- Peak GPU temperature: __________ °C

## Notes & Troubleshooting

### If inference is slow:
1. Check TorchScript compilation succeeded (check logs)
2. Verify FP16 is actually running: `watch nvidia-smi` during inference
3. Profile with `torch.utils.bottleneck`: `python -m torch.utils.bottleneck app.py`
4. Reduce SEQ_LEN or INPUT_SIZE if latency is critical
5. Enable quantization (INT8) for additional 2-4x speedup

### If GPU memory usage is high:
1. Disable TorchScript (use FP16 directly)
2. Reduce HIDDEN_SIZE from 256 to 128
3. Use `torch.cuda.empty_cache()` between batches
4. Move to SQLite to reduce database overhead

### If results aren't saving:
1. Verify database connection on Jetson
2. Check user authentication with `/api/user-info`
3. Review `save_progress()` and `flush_progress()` logs
4. Ensure `source` parameter is passed with every prediction

## Sign-Off

Deployment Date: _______________
Deployed By: _______________
Performance Validated: ___ Yes ___ No
Results Tracking Working: ___ Yes ___ No
Ready for Production: ___ Yes ___ No
