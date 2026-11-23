module.exports = {
  apps: [
    {
      name: "ml-services",
      cwd: "/var/www/ciis-peshawar/ml_services",
      script: "api_server.py",
      interpreter: "/var/www/ciis-peshawar/ml_services/venv/bin/python3",
      env: {
        ML_API_HOST: "0.0.0.0",
        ML_API_PORT: "8100",
        ML_DEVICE: "0",
        // Pin to host GPU 0 (gpu1 process pins GPU 1) so both never land on the same card.
        CUDA_VISIBLE_DEVICES: "0",
        ML_PREFER_GPU: "true",
        ML_GPU_CPU_FALLBACK: "true",
        // Per-process identity counter file + motion clip folder (data/identity_counters.gpu0.json,
        // motion_clips/gpu0/), and interleaved GP/GV/GO numbers so the two nodes never hand
        // out the same code (gpu0 = even, gpu1 = odd).
        ML_NODE_ID: "gpu0",
        ML_IDENTITY_CODE_STRIDE: "2",
        ML_IDENTITY_CODE_OFFSET: "0",
        // Set the same secret in the backend (sent as header X-ML-API-Key). Unset = open API.
        // ML_API_KEY: "",
      },
      autorestart: true,
      max_restarts: 10,
      restart_delay: 3000,
      watch: false,
    },
  ],
};
