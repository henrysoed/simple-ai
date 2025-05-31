# 🚀 Cara Deploy Neural Network ke Vercel

## Langkah-langkah Deploy:

### 1. Persiapan Repository Git
```bash
# Di folder project Anda
git init
git add .
git commit -m "Initial commit: Neural Network Demo"
```

### 2. Upload ke GitHub
1. Buat repository baru di GitHub (public/private)
2. Connect local repository ke GitHub:
```bash
git remote add origin https://github.com/username/neural-network-demo.git
git branch -M main
git push -u origin main
```

### 3. Deploy ke Vercel

#### Opsi A: Via Vercel Dashboard (Mudah)
1. Buka [vercel.com](https://vercel.com)
2. Sign up/Login dengan GitHub
3. Klik "New Project"  
4. Import repository GitHub Anda
5. Vercel akan otomatis detect Python project
6. Klik "Deploy"
7. Tunggu proses deployment (2-3 menit)
8. Aplikasi akan live di URL Vercel!

#### Opsi B: Via Vercel CLI
```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy
vercel

# Follow prompts dan aplikasi akan live!
```

### 4. Testing
Setelah deploy berhasil, Anda akan mendapat URL seperti:
`https://your-project-name.vercel.app`

Test fitur-fitur:
- ✅ Training XOR network
- ✅ Activation function comparison  
- ✅ Real-time neural network demo

### 5. Custom Domain (Optional)
Di Vercel dashboard → Project → Settings → Domains
Tambahkan domain custom Anda.

## 🛠️ Troubleshooting

### Error: Module not found
- Pastikan semua dependencies ada di `requirements.txt`
- Check bahwa `neural_network.py` ada di folder `api/`

### Error: Function timeout
- Vercel free plan limit 10 detik execution time
- Sudah dioptimasi untuk batasan ini di kode

### Error: Memory limit
- Vercel free plan: 1024MB memory
- Neural network ringan, should be fine

## 📁 File Structure untuk Vercel
```
neural-network-demo/
├── api/
│   ├── index.py          # ✅ Main serverless function
│   └── neural_network.py # ✅ ML logic
├── templates/
│   └── index.html        # ✅ Web interface  
├── vercel.json           # ✅ Vercel config
├── requirements.txt      # ✅ Dependencies
└── .vercelignore        # ✅ Ignore files
```

## 🎯 Hasil Akhir
Setelah deploy berhasil, Anda akan punya:
- ✅ Web app neural network yang bisa diakses online
- ✅ Interface interaktif untuk training/testing
- ✅ API endpoints untuk neural network operations
- ✅ Responsive design yang bekerja di mobile/desktop
- ✅ Gratis hosting di Vercel!

## 🔄 Update Aplikasi
Untuk update aplikasi:
1. Edit kode di local
2. Commit & push ke GitHub
3. Vercel otomatis re-deploy (auto deployment)

Selamat! Neural network Anda sekarang online! 🎉
