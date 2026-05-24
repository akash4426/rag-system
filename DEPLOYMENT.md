# 🚀 Production Deployment Guide

## Render Backend Deployment

### Step 1: Prepare Your Repository
```bash
# Ensure .env is not committed
echo ".env" >> .gitignore

# Push to GitHub
git add .
git commit -m "Production-ready: cleaned unnecessary files"
git push origin main
```

### Step 2: Deploy to Render
1. Go to [render.com](https://render.com)
2. Create a new **Web Service**
3. Connect your GitHub repository
4. Configure:
   - **Environment:** Docker
   - **Plan:** Standard (starter if low traffic)
   - **Build Command:** Automatic (uses Dockerfile)
   - **Start Command:** Automatic

### Step 3: Set Environment Variables
In Render dashboard → Environment:
- `PINECONE_API_KEY` - Your Pinecone API key
- `OPENAI_API_KEY` - Your OpenAI API key
- `PINECONE_INDEX_NAME` - `rag-system` (or your custom name)
- `PINECONE_ENVIRONMENT` - `us-east-1`
- `ALLOWED_ORIGINS` - (Set after Vercel deployment)

### Step 4: Get Backend URL
After deployment, Render provides a URL like:
```
https://your-service-name.onrender.com
```

---

## Vercel Frontend Deployment

### Step 1: Build Frontend
```bash
cd ui
npm install
npm run build
```

### Step 2: Deploy to Vercel
1. Go to [vercel.com](https://vercel.com)
2. Import project from GitHub
3. Configure:
   - **Framework Preset:** Vite
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
   - **Install Command:** `npm install`

### Step 3: Environment Variables (Optional)
For advanced setup, in Vercel dashboard:
- `VITE_API_URL` - Points to `https://your-service-name.onrender.com` (optional - will use relative URLs by default)

### Step 4: Deployment
Push to GitHub and Vercel automatically deploys:
```bash
git push origin main
```

---

## Post-Deployment: Update CORS

After both deployments are live:

1. Go to Render dashboard
2. Edit Environment Variables
3. Update `ALLOWED_ORIGINS`:
   ```
   https://your-project-name.vercel.app,http://localhost:5173
   ```
4. Restart the service

---

## Testing Production

### Health Check
```bash
curl https://your-service-name.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "vector_db": "pinecone"
}
```

### Test Upload
```bash
curl -F "file=@test.txt" https://your-service-name.onrender.com/upload
```

### Test Query
```bash
curl -X POST https://your-service-name.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "session_id": "test-session"}'
```

---

## Production Checklist

- [x] All development files removed
- [x] Docker configured correctly
- [x] Environment variables prepared
- [x] CORS configured for production
- [x] Pinecone API key set
- [x] OpenAI API key set
- [x] Render deployment configured
- [x] Vercel deployment configured
- [x] Health endpoint verified
- [x] End-to-end testing completed

---

## Monitoring & Maintenance

### Logs
- **Render:** Dashboard → Logs tab
- **Vercel:** Dashboard → Deployments → Logs

### Common Issues

**Issue:** 502 Bad Gateway
- Check Render logs for backend errors
- Verify environment variables are set
- Restart the service

**Issue:** CORS Errors in Frontend
- Verify `ALLOWED_ORIGINS` includes your Vercel domain
- Check browser console for exact error
- Restart Render service after updating CORS

**Issue:** Pinecone Connection Failed
- Verify `PINECONE_API_KEY` is correct
- Check Pinecone dashboard for active keys
- Confirm `PINECONE_INDEX_NAME` exists

---

## Performance Tips

1. **Caching:** Consider adding Redis for query result caching
2. **Rate Limiting:** Add rate limiting middleware for API endpoints
3. **Monitoring:** Set up alerts for 5xx errors
4. **Database:** Monitor Pinecone vector count and query latency

---

## Next Steps

Once deployed:
1. Share Vercel frontend URL with users
2. Monitor logs for errors
3. Scale plan if needed (upgrade on Render/Vercel)
4. Set up backup/restore procedures for Pinecone index
