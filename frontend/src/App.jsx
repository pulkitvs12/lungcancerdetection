import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { 
  UploadCloud, 
  FileImage, 
  AlertCircle, 
  RefreshCw, 
  Activity, 
  Info, 
  HelpCircle, 
  BookOpen, 
  Layers, 
  ShieldCheck, 
  ShieldAlert, 
  Microscope, 
  Stethoscope, 
  Dna, 
  BarChart2, 
  CheckCircle2, 
  ChevronDown, 
  ChevronUp,
  Cpu,
  ArrowRight
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

// Medical knowledge repository for the 4 histological categories
const CLASS_INFO = {
  adenocarcinoma: {
    fullName: "Lung Adenocarcinoma (LUAD)",
    subtype: "Non-Small Cell Lung Cancer (NSCLC)",
    prevalence: "~40% of all primary lung cancers (Most Prevalent)",
    origin: "Peripheral mucus-secreting glandular epithelial cells of the alveoli and small bronchi.",
    significance: "The most common form of lung cancer. Unlike squamous cell carcinoma, it frequently arises in peripheral regions of the lungs and is the most common lung cancer seen in women, younger adults, and non-smokers (though still strongly associated with smoking).",
    histologicalFeatures: "Formations of glandular or acinar structures, mucin secretion within cellular vacuoles, papillary/micropapillary clusters, and lepidic growth pattern tracking intact alveolar septa.",
    diagnosticNextSteps: [
      "Immunohistochemistry (IHC): Confirmed via TTF-1 (+), Napsin A (+), and p40 (-).",
      "Comprehensive Molecular Profiling: Testing for actionable driver mutations (EGFR, ALK translocations, ROS1, KRAS G12C, BRAF V600E, and PD-L1 expression) to guide targeted tyrosine kinase inhibitor (TKI) therapies.",
      "Contrast-enhanced Chest/Abdominal CT & Whole-Body PET/CT for TNM staging."
    ],
    color: "#38bdf8"
  },
  squamous_cell_carcinoma: {
    fullName: "Squamous Cell Carcinoma (LUSC)",
    subtype: "Non-Small Cell Lung Cancer (NSCLC)",
    prevalence: "~25% - 30% of all lung malignancies",
    origin: "Flat squamous epithelial cells lining the proximal and central segmental bronchi.",
    significance: "Demonstrates the strongest epidemiological link to chronic tobacco smoke exposure. Because it arises centrally near the main tracheobronchial tree, patients often present with early symptomatic bronchial obstruction, persistent cough, atelectasis, or hemoptysis (coughing blood).",
    histologicalFeatures: "Keratin pearl formation, intercellular desmosomal bridges, individual cell keratinization with dense eosinophilic cytoplasm, and frequent central cavitation/necrosis on imaging.",
    diagnosticNextSteps: [
      "Immunohistochemistry (IHC): Confirmed via p40 (+), p63 (+), CK5/6 (+), and TTF-1 (-).",
      "Flexible Bronchoscopy: Direct visualization and endobronchial biopsy of the central obstructing mass.",
      "Therapeutic Note: Anti-angiogenic therapies (e.g. bevacizumab) are generally contraindicated due to severe risks of central pulmonary hemorrhage."
    ],
    color: "#818cf8"
  },
  large_cell_carcinoma: {
    fullName: "Large Cell Carcinoma (LCC)",
    subtype: "Non-Small Cell Lung Cancer (NSCLC - Undifferentiated)",
    prevalence: "~10% - 15% of lung malignancies",
    origin: "Transformed pulmonary epithelial cells that lack distinct morphological differentiation.",
    significance: "A diagnosis of exclusion characterized by high biological aggressiveness, rapid cellular doubling time, and an early propensity for hematogenous dissemination (especially to the brain, liver, adrenal glands, and bones).",
    histologicalFeatures: "Sheets or solid nests of large, atypical polygonal cells with vesicular nuclei, prominent prominent nucleoli, abundant cytoplasm, and no architectural features of glandular or squamous lineage.",
    diagnosticNextSteps: [
      "Broad IHC Differentiation Panel: Extensive staining (TTF-1, p40, chromogranin, synaptophysin) to rule out poorly differentiated LUAD, LUSC, or neuroendocrine carcinoma.",
      "Systemic Staging: Brain MRI with IV contrast and PET-CT staging is urgent due to high metastatic frequency.",
      "Evaluation for Multimodal Chemotherapy/Immunotherapy regimens."
    ],
    color: "#f59e0b"
  },
  normal: {
    fullName: "Normal Pulmonary Parenchyma (Benign)",
    subtype: "Healthy / Non-Neoplastic Lung Tissue",
    prevalence: "Baseline Physiological Control",
    origin: "Healthy respiratory parenchyma consisting of alveolar ducts, sacs, and terminal bronchioles.",
    significance: "No cytological atypia, architectural disorganization, or neoplastic cellular clusters are detected in the scanned field. The tissue exhibits normal cellular morphology and preservation of the physiological blood-air barrier.",
    histologicalFeatures: "Thin, delicate alveolar walls lined by Type I and Type II pneumocytes, clear alveolar spaces, absence of nuclear pleomorphism, and absence of atypical mitoses.",
    diagnosticNextSteps: [
      "Standard routine clinical monitoring.",
      "If patient has active respiratory symptoms (e.g. chronic cough, shortness of breath), evaluate non-malignant etiologies such as infection (pneumonia), COPD, interstitial lung disease, or asthma.",
      "Correlate with complete thoracic radiology to ensure whole-organ assessment."
    ],
    color: "#10b981"
  }
};

// Ensemble Model Benchmark Data
const ENSEMBLE_MODELS = [
  {
    name: "ResNet50",
    architecture: "Deep Residual Network (50 Layers)",
    accuracy: "84.4%",
    precision: "84.7%",
    recall: "84.4%",
    f1: "84.4%",
    keyStrength: "Residual skip connections preserve gradients across 50 deep layers, enabling state-of-the-art hierarchical texture extraction."
  },
  {
    name: "InceptionV3",
    architecture: "Multi-Scale Factorized Convolutions",
    accuracy: "71.1%",
    precision: "73.9%",
    recall: "71.1%",
    f1: "69.9%",
    keyStrength: "Simultaneous 1x1, 3x3, and 5x5 filters capture both fine intracellular nuances and broad macroscopic alveolar lesions."
  },
  {
    name: "MobileNetV2",
    architecture: "Inverted Residuals & Linear Bottlenecks",
    accuracy: "67.6%",
    precision: "71.8%",
    recall: "67.6%",
    f1: "67.4%",
    keyStrength: "Depthwise separable convolutions minimize parameter overhead while prioritizing salient spatial boundaries."
  },
  {
    name: "DenseNet121",
    architecture: "Densely Connected Convolutional Networks",
    accuracy: "56.5%",
    precision: "67.0%",
    recall: "56.5%",
    f1: "51.4%",
    keyStrength: "Direct connections between all layers maximize feature reuse and gradient flow across deep representations."
  }
];

function App() {
  const [activeTab, setActiveTab] = useState('diagnostics'); // 'diagnostics' | 'overview' | 'pathology' | 'interpretation'
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState({ online: false, checking: true });
  const [selectedChartModel, setSelectedChartModel] = useState('average'); // 'average' | model name
  const [showInterpretationDetails, setShowInterpretationDetails] = useState(true);
  
  const fileInputRef = useRef(null);

  // Poll backend health status on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await axios.get(`${API_BASE_URL}/health`, { timeout: 3000 });
        if (res.data?.status === 'healthy') {
          setBackendStatus({ online: true, checking: false });
        } else {
          setBackendStatus({ online: false, checking: false });
        }
      } catch (e) {
        console.warn("Backend health check failed:", e);
        setBackendStatus({ online: false, checking: false });
      }
    };
    checkHealth();
  }, []);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (selectedFile) => {
    setError(null);
    if (!selectedFile.type.startsWith('image/')) {
      setError("Please upload a valid image file (JPEG, PNG, WEBP, or TIFF).");
      return;
    }
    if (selectedFile.size > 15 * 1024 * 1024) {
      setError("File size exceeds 15MB limit.");
      return;
    }
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResults(null);
  };

  const analyzeImage = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResults(response.data);
      // Auto-switch to ensemble average or ResNet50
      setSelectedChartModel('average');
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "Failed to analyze image. Ensure the FastAPI backend server is running.");
    } finally {
      setLoading(false);
    }
  };

  // Generate a synthetic test lung pattern for quick demonstration without user file upload
  const loadSampleImage = () => {
    const canvas = document.createElement('canvas');
    canvas.width = 300;
    canvas.height = 300;
    const ctx = canvas.getContext('2d');
    
    // Draw simulated histopathology cellular texture
    ctx.fillStyle = '#2d1b4e';
    ctx.fillRect(0, 0, 300, 300);
    
    // Alveolar and cellular rings
    for (let i = 0; i < 40; i++) {
      ctx.beginPath();
      const x = Math.random() * 300;
      const y = Math.random() * 300;
      const r = 8 + Math.random() * 25;
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${180 + Math.random() * 50}, ${80 + Math.random() * 60}, ${180 + Math.random() * 60}, 0.55)`;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#c084fc';
      ctx.stroke();
    }
    
    canvas.toBlob((blob) => {
      const sampleFile = new File([blob], "sample_histopathology_scan.png", { type: "image/png" });
      handleFile(sampleFile);
    }, 'image/png');
  };

  const reset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResults(null);
    setError(null);
  };

  // Transform probabilities for Recharts with custom support for average or single model
  const getChartData = () => {
    if (!results?.models) return [];
    
    if (selectedChartModel === 'average') {
      const classes = ['adenocarcinoma', 'large_cell_carcinoma', 'squamous_cell_carcinoma', 'normal'];
      return classes.map(cls => {
        let sum = 0;
        let count = 0;
        Object.values(results.models).forEach(m => {
          if (m.probabilities[cls] !== undefined) {
            sum += m.probabilities[cls];
            count += 1;
          }
        });
        const avg = count > 0 ? (sum / count) * 100 : 0;
        return {
          key: cls,
          name: cls.replace(/_/g, ' '),
          prob: Number(avg.toFixed(1))
        };
      });
    }

    const modelData = results.models[selectedChartModel];
    if (!modelData) return [];
    
    return Object.entries(modelData.probabilities).map(([cls, val]) => ({
      key: cls,
      name: cls.replace(/_/g, ' '),
      prob: Number((val * 100).toFixed(1))
    }));
  };

  // Determine agreement level interpretation
  const getConsensusInterpretation = () => {
    if (!results) return null;
    const { votes } = results.majority_vote;
    
    if (votes === 4) {
      return {
        level: "Unanimous Consensus (4/4 Models)",
        badgeClass: "high",
        explanation: "All 4 deep learning architectures independently produced the exact same diagnosis. This reflects strong, unambiguous morphological agreement across multi-scale feature representations.",
        recommendation: "High confidence in feature classification. Proceed with histological validation, IHC biomarker staging, and clinical correlation."
      };
    } else if (votes === 3) {
      return {
        level: "Strong Majority Consensus (3/4 Models)",
        badgeClass: "medium",
        explanation: "3 out of 4 models agree on the primary diagnosis, with 1 model exhibiting a divergent vote. This frequently occurs in borderline cellular atypia or mixed histopathological morphologies.",
        recommendation: "Review the secondary model's probability distribution to check for overlapping features between adenocarcinoma and squamous or large cell subtypes."
      };
    } else {
      return {
        level: "Split / Discordant Ensemble Vote (2/4 Models)",
        badgeClass: "split",
        explanation: "The ensemble is split, with equal or fragmented votes among architectures. This signifies complex tissue heterogeneity, poorly differentiated cells, or potential image artifacts.",
        recommendation: "Plurality vote should be interpreted with high caution. Mandatory pathologist slide review, repeated slice imaging, and extended IHC panel are advised."
      };
    }
  };

  const consensus = getConsensusInterpretation();
  const predictedClassData = results ? CLASS_INFO[results.majority_vote.prediction] : null;

  return (
    <div className="app-container">
      {/* Header & Brand Navigation */}
      <header className="header">
        <div className="header-top">
          <div className="brand-section">
            <div className="brand-icon-box">
              <Microscope size={26} />
            </div>
            <div>
              <h1 className="brand-title">Lung Cancer Detection</h1>
              <p className="brand-tagline">Deep Learning Multi-Model Ensemble for Histopathology & CT Analysis</p>
            </div>
          </div>

          <div className="status-badge">
            <span className={`status-dot ${backendStatus.online ? 'online' : ''}`}></span>
            <span>
              {backendStatus.online 
                ? "Backend Ready • 4 Models Active" 
                : backendStatus.checking 
                  ? "Connecting to Models..." 
                  : "Backend Offline (Port 8001)"}
            </span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="nav-tabs">
          <button 
            className={`nav-tab-btn ${activeTab === 'diagnostics' ? 'active' : ''}`}
            onClick={() => setActiveTab('diagnostics')}
          >
            <Activity size={18} /> Diagnostic Workspace
          </button>
          <button 
            className={`nav-tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            <Layers size={18} /> Project Overview & Ensemble AI
          </button>
          <button 
            className={`nav-tab-btn ${activeTab === 'pathology' ? 'active' : ''}`}
            onClick={() => setActiveTab('pathology')}
          >
            <BookOpen size={18} /> Pathology Reference Library
          </button>
          <button 
            className={`nav-tab-btn ${activeTab === 'interpretation' ? 'active' : ''}`}
            onClick={() => setActiveTab('interpretation')}
          >
            <HelpCircle size={18} /> How to Interpret Results
          </button>
        </nav>
      </header>

      {/* TAB 1: DIAGNOSTICS WORKSPACE */}
      {activeTab === 'diagnostics' && (
        <>
          {!results && (
            <div className="card">
              <div style={{ marginBottom: '1.5rem', textAlign: 'center' }}>
                <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.6rem', color: '#fff', marginBottom: '0.4rem' }}>
                  AI-Powered Pulmonary Image Classification
                </h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', maxWidth: '750px', margin: '0 auto' }}>
                  Upload a lung CT or histopathology slice. Our ensemble of 4 convolutional neural networks 
                  (<strong>ResNet50</strong>, <strong>InceptionV3</strong>, <strong>MobileNetV2</strong>, and <strong>DenseNet121</strong>) 
                  will process the image concurrently to evaluate malignancy and determine histological consensus.
                </p>
              </div>

              <div 
                className={`upload-panel ${isDragging ? 'drag-active' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current.click()}
              >
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleFileSelect} 
                  accept="image/*"
                />
                
                {previewUrl ? (
                  <div>
                    <img src={previewUrl} alt="Preview" className="preview-img" style={{ maxHeight: '220px', borderRadius: '8px' }} />
                    <p style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', color: 'var(--accent-cyan)', fontWeight: 600 }}>
                      <FileImage size={20} /> {file?.name || "Uploaded Image"}
                    </p>
                    <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                      Click or drag a different image to replace
                    </p>
                  </div>
                ) : (
                  <div>
                    <div className="upload-icon-circle">
                      <UploadCloud size={34} />
                    </div>
                    <h3>Drag & Drop Lung CT / Histopathology Image</h3>
                    <p>Supported standard formats: JPEG, PNG, WEBP, TIFF (Max 15MB)</p>
                    <div className="upload-guidelines">
                      <span className="guideline-tag"><CheckCircle2 size={13} color="var(--success)" /> Chest CT Axial Slices</span>
                      <span className="guideline-tag"><CheckCircle2 size={13} color="var(--success)" /> H&E Histopathology</span>
                      <span className="guideline-tag"><CheckCircle2 size={13} color="var(--success)" /> 4-Model Concurrent Inference</span>
                    </div>
                  </div>
                )}
              </div>
              
              {error && (
                <div style={{ color: 'var(--danger)', marginTop: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem', justifyContent: 'center', background: 'var(--danger-bg)', padding: '0.75rem', borderRadius: '8px' }}>
                  <AlertCircle size={20} /> {error}
                </div>
              )}

              <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', marginTop: '1.5rem', flexWrap: 'wrap' }}>
                {file && !loading && (
                  <button className="btn-primary" onClick={analyzeImage}>
                    <Activity size={18} /> Run Ensemble Diagnostic
                  </button>
                )}

                {!file && (
                  <button className="btn-secondary" onClick={loadSampleImage}>
                    <Microscope size={18} /> Test with Sample Slide
                  </button>
                )}

                {file && !loading && (
                  <button className="btn-secondary" onClick={reset}>
                    Clear Selection
                  </button>
                )}
              </div>

              {loading && (
                <div className="loader-container">
                  <div className="loader"></div>
                  <div style={{ textAlign: 'center' }}>
                    <p style={{ color: '#fff', fontWeight: 600, fontSize: '1.1rem' }}>Running 4-Model Inference...</p>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.35rem' }}>
                      Evaluating ResNet50, MobileNetV2, InceptionV3, and DenseNet121 feature vectors
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* RESULTS DISPLAY */}
          {results && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
              {/* Top Action Bar */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
                <div>
                  <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.75rem', color: '#fff' }}>Diagnostic Results & Clinical Interpretation</h2>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Comprehensive 4-Model ensemble assessment and subtype breakdown</p>
                </div>
                <button className="btn-primary" onClick={reset} style={{ margin: 0 }}>
                  <RefreshCw size={16} /> Analyze Another Scan
                </button>
              </div>

              {/* Consensus & Primary Diagnosis Box */}
              <div className="results-summary-grid">
                <div className="card consensus-box">
                  <div>
                    <div className="card-title">
                      <ShieldCheck size={20} /> Primary Ensemble Consensus
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.5rem' }}>
                      <span className={`consensus-header-badge ${results.is_cancer ? 'cancer' : 'normal'}`}>
                        {results.is_cancer ? <ShieldAlert size={15} /> : <ShieldCheck size={15} />}
                        {results.is_cancer ? "Malignancy Detected" : "Benign / Normal Tissue"}
                      </span>
                    </div>

                    <h3 className={`consensus-prediction-title ${results.is_cancer ? 'cancer' : 'normal'}`}>
                      {predictedClassData?.fullName || results.majority_vote.prediction.replace(/_/g, ' ')}
                    </h3>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
                      {predictedClassData?.subtype}
                    </p>
                  </div>

                  {/* Consensus Agreement Meter */}
                  <div className="agreement-meter-wrapper">
                    <div className="meter-header">
                      <span className="meter-header-label">Ensemble Concordance</span>
                      <span className="meter-header-value">
                        {results.majority_vote.votes} of {results.majority_vote.total_models} Models Agreed ({(results.majority_vote.agreement * 100).toFixed(0)}%)
                      </span>
                    </div>
                    <div className="meter-bar-track">
                      <div 
                        className={`meter-bar-fill ${consensus?.badgeClass || 'high'}`} 
                        style={{ width: `${results.majority_vote.agreement * 100}%` }}
                      ></div>
                    </div>
                    <p className="meter-interpretation-text">
                      <strong>{consensus?.level}:</strong> {consensus?.explanation}
                    </p>
                  </div>
                </div>

                {/* Uploaded Scan Card */}
                <div className="card result-image-box">
                  <div className="card-title" style={{ width: '100%' }}>
                    <FileImage size={18} /> Analyzed Slide / Scan
                  </div>
                  <img src={previewUrl} alt="Analyzed Scan" className="result-image-frame" />
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.75rem' }}>
                    {file?.name} • Evaluated at model target dimensions
                  </span>
                </div>
              </div>

              {/* DYNAMIC SIGNIFICANCE & INTERPRETATION SECTION */}
              <div className="significance-card">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.5rem' }}>
                  <Stethoscope size={22} color="var(--accent-cyan)" />
                  <h3 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.35rem', color: '#fff' }}>
                    What This Result Signifies: {predictedClassData?.fullName}
                  </h3>
                </div>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', lineHeight: '1.6' }}>
                  {predictedClassData?.significance}
                </p>

                <div className="significance-grid">
                  <div className="significance-col">
                    <h4><Microscope size={17} /> Histopathological Characteristics</h4>
                    <p>{predictedClassData?.histologicalFeatures}</p>
                    <p style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
                      <strong>Cellular Origin:</strong> {predictedClassData?.origin}
                    </p>
                  </div>

                  <div className="significance-col">
                    <h4><Dna size={17} /> Clinical Recommendations & Follow-Up</h4>
                    <ul>
                      {predictedClassData?.diagnosticNextSteps.map((step, idx) => (
                        <li key={idx}>
                          <ArrowRight size={14} />
                          <span>{step}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>

              {/* Model Comparison Grid */}
              <div>
                <h3 className="card-title">
                  <Cpu size={20} /> Individual Neural Network Assessments
                </h3>
                <div className="models-grid">
                  {Object.entries(results.models).map(([modelName, data]) => {
                    const isWinner = data.prediction === results.majority_vote.prediction;
                    const isCancerPred = data.prediction !== 'normal';
                    return (
                      <div className={`model-card ${isWinner ? 'winner-model' : ''}`} key={modelName}>
                        <div className="model-card-header">
                          <span className="model-name">{modelName}</span>
                          <span className="model-conf-badge">
                            {(data.confidence * 100).toFixed(1)}% Confidence
                          </span>
                        </div>

                        <div className={`model-pred-pill ${isCancerPred ? 'cancer' : 'normal'}`}>
                          {isCancerPred ? <ShieldAlert size={16} /> : <ShieldCheck size={16} />}
                          <span>{data.prediction.replace(/_/g, ' ')}</span>
                        </div>

                        <div className="prob-list">
                          {Object.entries(data.probabilities).map(([cls, prob]) => {
                            const isTopForThisModel = cls === data.prediction;
                            return (
                              <div className="prob-row" key={cls}>
                                <div className="prob-meta">
                                  <span className={`prob-name ${isTopForThisModel ? 'winner' : ''}`}>
                                    {cls.replace(/_/g, ' ')}
                                  </span>
                                  <span className={`prob-val ${isTopForThisModel ? 'winner' : ''}`}>
                                    {(prob * 100).toFixed(1)}%
                                  </span>
                                </div>
                                <div className="prob-track">
                                  <div 
                                    className="prob-bar"
                                    style={{ 
                                      width: `${prob * 100}%`,
                                      backgroundColor: cls === 'normal' 
                                        ? 'var(--success)' 
                                        : (isTopForThisModel ? 'var(--accent-cyan)' : 'var(--text-dim)')
                                    }}
                                  ></div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Interactive Probability Distribution Chart */}
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.25rem' }}>
                  <div className="card-title" style={{ margin: 0, border: 'none', padding: 0 }}>
                    <BarChart2 size={20} /> Class Probability Distribution Spectrum
                  </div>

                  {/* Chart Model Toggle Buttons */}
                  <div className="chart-controls">
                    <button 
                      className={`chart-model-btn ${selectedChartModel === 'average' ? 'active' : ''}`}
                      onClick={() => setSelectedChartModel('average')}
                    >
                      Ensemble Mean
                    </button>
                    {Object.keys(results.models).map(mName => (
                      <button 
                        key={mName}
                        className={`chart-model-btn ${selectedChartModel === mName ? 'active' : ''}`}
                        onClick={() => setSelectedChartModel(mName)}
                      >
                        {mName}
                      </button>
                    ))}
                  </div>
                </div>

                <div style={{ height: '320px', width: '100%', marginTop: '0.5rem' }}>
                  <ResponsiveContainer>
                    <BarChart data={getChartData()} margin={{ top: 20, right: 30, left: 10, bottom: 25 }}>
                      <XAxis 
                        dataKey="name" 
                        stroke="var(--text-muted)" 
                        fontSize={13} 
                        tickLine={false}
                        dy={8}
                      />
                      <YAxis 
                        stroke="var(--text-muted)" 
                        fontSize={12} 
                        domain={[0, 100]}
                        tickFormatter={(v) => `${v}%`}
                        tickLine={false}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#0f172a', 
                          borderColor: 'rgba(56, 189, 248, 0.3)', 
                          borderRadius: '8px',
                          color: '#fff',
                          boxShadow: '0 8px 16px rgba(0,0,0,0.5)'
                        }} 
                        formatter={(val) => [`${val}%`, 'Calculated Probability']}
                      />
                      <Bar dataKey="prob" radius={[6, 6, 0, 0]}>
                        {getChartData().map((entry) => (
                          <Cell 
                            key={`cell-${entry.key}`} 
                            fill={entry.key === 'normal' ? 'var(--success)' : (CLASS_INFO[entry.key]?.color || 'var(--accent-cyan)')} 
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <p style={{ textAlign: 'center', fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                  Displaying probability distribution for: <strong>{selectedChartModel === 'average' ? 'Ensemble Combined Mean' : selectedChartModel}</strong>. Select buttons above to inspect individual models.
                </p>
              </div>

              {/* Accordion / Collapsible Guide on How to Interpret */}
              <div className="accordion-box">
                <div 
                  className="accordion-header"
                  onClick={() => setShowInterpretationDetails(!showInterpretationDetails)}
                >
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Info size={18} color="var(--accent-cyan)" /> How Should Clinicians & Researchers Interpret These Numbers?
                  </span>
                  {showInterpretationDetails ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                </div>

                {showInterpretationDetails && (
                  <div className="accordion-body">
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem' }}>
                      <div>
                        <h4 style={{ color: '#fff', marginBottom: '0.4rem', fontSize: '0.95rem' }}>1. Softmax Confidence vs Diagnostic Certainty</h4>
                        <p>
                          A model assigning a 95% confidence score indicates high activation in its trained convolutional filters, not absolute medical ground truth. Real-world histopathology can have tissue fixation artifacts or necrotic regions that skew individual scores.
                        </p>
                      </div>
                      <div>
                        <h4 style={{ color: '#fff', marginBottom: '0.4rem', fontSize: '0.95rem' }}>2. Why Models Might Disagree</h4>
                        <p>
                          <strong>ResNet50</strong> evaluates deep hierarchical texture patterns; <strong>InceptionV3</strong> assesses multi-scale structures simultaneously; <strong>DenseNet121</strong> reuses iterative features. Divergence often points to mixed adenosquamous cellularity or poorly differentiated areas.
                        </p>
                      </div>
                      <div>
                        <h4 style={{ color: '#fff', marginBottom: '0.4rem', fontSize: '0.95rem' }}>3. Clinical Action Guideline</h4>
                        <p>
                          Never base treatment or staging decisions exclusively on an AI prediction. Use this tool for rapid triage, secondary review verification, and identifying complex borderline cases that require immediate immunohistochemistry (IHC).
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      )}

      {/* TAB 2: PROJECT OVERVIEW & ENSEMBLE AI */}
      {activeTab === 'overview' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <div className="overview-hero">
            <h2>About the Lung Cancer Detection Project</h2>
            <p>
              Lung cancer remains the leading cause of oncological mortality worldwide, largely due to late-stage discovery 
              and complex histopathological differentiation. This system implements an advanced 
              <strong> 4-model Convolutional Neural Network (CNN) Ensemble</strong> designed to assist clinicians, 
              pathologists, and oncological researchers in early detection and subtype stratification of thoracic images.
            </p>

            <div className="overview-features-grid">
              <div className="feature-pill">
                <div className="feature-icon-wrapper"><Layers size={20} /></div>
                <div>
                  <h4>Multi-Model Ensemble</h4>
                  <p>Eliminates single-model blind spots by aggregating 4 diverse deep learning architectures.</p>
                </div>
              </div>

              <div className="feature-pill">
                <div className="feature-icon-wrapper"><ShieldCheck size={20} /></div>
                <div>
                  <h4>Democratic Majority Voting</h4>
                  <p>Synthesizes predictions to maximize diagnostic precision and quantify model agreement.</p>
                </div>
              </div>

              <div className="feature-pill">
                <div className="feature-icon-wrapper"><Microscope size={20} /></div>
                <div>
                  <h4>4-Class Histological Scope</h4>
                  <p>Differentiates Adenocarcinoma, Squamous Cell, Large Cell, and Normal lung tissue.</p>
                </div>
              </div>

              <div className="feature-pill">
                <div className="feature-icon-wrapper"><Stethoscope size={20} /></div>
                <div>
                  <h4>Decision Support Protocol</h4>
                  <p>Provides actionable clinical context, IHC biomarker pathways, and staging next steps.</p>
                </div>
              </div>
            </div>
          </div>

          {/* Model Architecture Deep Dive */}
          <div className="card">
            <h3 className="card-title"><Cpu size={20} /> The 4 Neural Network Architectures</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', marginBottom: '1.5rem' }}>
              Each architecture incorporates a fundamentally distinct spatial learning philosophy. By training these models 
              independently on thoracic imaging datasets, the ensemble achieves complementary feature extraction:
            </p>

            <div className="models-grid">
              {ENSEMBLE_MODELS.map(m => (
                <div className="model-card" key={m.name}>
                  <div className="model-card-header">
                    <span className="model-name">{m.name}</span>
                    <span className="model-conf-badge">{m.accuracy} Accuracy</span>
                  </div>
                  <p style={{ fontSize: '0.85rem', color: '#fff', fontWeight: 600, marginBottom: '0.5rem' }}>
                    {m.architecture}
                  </p>
                  <p style={{ fontSize: '0.83rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                    {m.keyStrength}
                  </p>
                </div>
              ))}
            </div>

            {/* Academic Benchmark Comparison Table */}
            <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.15rem', color: '#fff', marginTop: '2rem', marginBottom: '0.75rem' }}>
              Academic Performance Comparison
            </h4>
            <div className="benchmark-table-container">
              <table className="benchmark-table">
                <thead>
                  <tr>
                    <th>Model Architecture</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                  </tr>
                </thead>
                <tbody>
                  {ENSEMBLE_MODELS.map(m => (
                    <tr key={m.name}>
                      <td style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>{m.name}</td>
                      <td className={parseFloat(m.accuracy) > 80 ? 'score-badge-high' : 'score-badge-med'}>{m.accuracy}</td>
                      <td>{m.precision}</td>
                      <td>{m.recall}</td>
                      <td>{m.f1}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: PATHOLOGY REFERENCE LIBRARY */}
      {activeTab === 'pathology' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <div className="card" style={{ background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9))' }}>
            <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.6rem', color: '#fff', marginBottom: '0.5rem' }}>
              Histopathological Classification & Diagnostic Guide
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', maxWidth: '850px' }}>
              Understanding the physiological distinction between histological subtypes is vital for interpreting model outputs. 
              Lung cancer is heterogeneous; different cellular origins dictate distinct surgical, chemotherapeutic, and targeted immunotherapy pathways.
            </p>
          </div>

          <div className="pathology-grid">
            {Object.entries(CLASS_INFO).map(([key, info]) => (
              <div className="pathology-card" key={key}>
                <div className="pathology-card-header">
                  <div className="pathology-title-wrap">
                    <h3>{info.fullName}</h3>
                    <span className="pathology-subtitle">{info.subtype}</span>
                  </div>
                  <span className="pathology-prevalence-badge">{info.prevalence}</span>
                </div>

                <div className="pathology-section">
                  <span className="pathology-section-title">Cellular Origin</span>
                  <p className="pathology-section-content">{info.origin}</p>
                </div>

                <div className="pathology-section">
                  <span className="pathology-section-title">Clinical Significance</span>
                  <p className="pathology-section-content">{info.significance}</p>
                </div>

                <div className="pathology-section">
                  <span className="pathology-section-title">Microscopic & Radiological Features</span>
                  <p className="pathology-section-content">{info.histologicalFeatures}</p>
                </div>

                <div className="pathology-section">
                  <span className="pathology-section-title">Diagnostic Pathways</span>
                  <div className="pathology-tags">
                    {info.diagnosticNextSteps.map((step, idx) => (
                      <span className="pathology-tag" key={idx}>{step.split(':')[0]}</span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* TAB 4: HOW TO INTERPRET RESULTS */}
      {activeTab === 'interpretation' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <div className="card">
            <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.6rem', color: '#fff', marginBottom: '0.5rem' }}>
              Framework for Interpreting AI Predictions
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', maxWidth: '850px' }}>
              Artificial intelligence algorithms in medical imaging serve as decision augmentation tools. 
              Follow this systematic 4-step framework to evaluate model predictions, assess confidence levels, 
              and determine clinical follow-up.
            </p>

            <div className="guide-steps-grid">
              <div className="guide-step-card">
                <div className="guide-step-num">1</div>
                <h4 className="guide-step-title">Check Majority Consensus</h4>
                <p className="guide-step-desc">
                  Look at the <strong>Agreement Percentage</strong>. If 4/4 models agree (100%), there is high architectural consensus. If agreement is 50-75%, carefully inspect which models diverged and check for mixed histological patterns.
                </p>
              </div>

              <div className="guide-step-card">
                <div className="guide-step-num">2</div>
                <h4 className="guide-step-title">Examine Probability Spectrum</h4>
                <p className="guide-step-desc">
                  Do not look only at the winner. Review the secondary (runner-up) probabilities. For instance, if Adenocarcinoma is 52% and Squamous Cell is 44%, the slide exhibits overlapping morphological features that warrant IHC staining.
                </p>
              </div>

              <div className="guide-step-card">
                <div className="guide-step-num">3</div>
                <h4 className="guide-step-title">Review Model Discrepancies</h4>
                <p className="guide-step-desc">
                  <strong>ResNet50</strong> has the highest baseline test accuracy (84.4%). If ResNet50 conflicts with DenseNet121, give greater initial consideration to ResNet50 while seeking pathologist clarification.
                </p>
              </div>

              <div className="guide-step-card">
                <div className="guide-step-num">4</div>
                <h4 className="guide-step-title">Correlate with Clinical Context</h4>
                <p className="guide-step-desc">
                  Cross-reference results with patient history (smoking pack-years, occupational exposures, genetic risk), anatomical lesion location (central vs peripheral), and confirmatory core needle biopsy.
                </p>
              </div>
            </div>
          </div>

          {/* Agreement Scale Explainer */}
          <div className="card">
            <h3 className="card-title"><ShieldCheck size={20} /> Understanding the Concordance Scale</h3>
            
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem', marginTop: '1rem' }}>
              <div style={{ background: 'rgba(16, 185, 129, 0.08)', border: '1px solid var(--success-border)', borderRadius: '10px', padding: '1.25rem' }}>
                <span style={{ color: 'var(--success)', fontWeight: 700, fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <CheckCircle2 size={18} /> 4/4 Models (100% Concordance)
                </span>
                <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                  <strong>Unanimous Consensus:</strong> All four deep neural network backbones extracted convergent feature representations. High algorithmic reliability for primary screening.
                </p>
              </div>

              <div style={{ background: 'rgba(245, 158, 11, 0.08)', border: '1px solid var(--warning-border)', borderRadius: '10px', padding: '1.25rem' }}>
                <span style={{ color: 'var(--warning)', fontWeight: 700, fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <AlertCircle size={18} /> 3/4 Models (75% Concordance)
                </span>
                <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                  <strong>Strong Majority:</strong> Three models agree; one is discordant. Often seen when cellular atypia or poor differentiation creates ambiguous receptive field boundaries.
                </p>
              </div>

              <div style={{ background: 'rgba(244, 63, 94, 0.08)', border: '1px solid var(--danger-border)', borderRadius: '10px', padding: '1.25rem' }}>
                <span style={{ color: 'var(--danger)', fontWeight: 700, fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <ShieldAlert size={18} /> 2/4 Models (50% Concordance)
                </span>
                <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                  <strong>Discordant / Split Vote:</strong> Models cannot reach decisive agreement. Indicates high tissue heterogeneity, mixed adenosquamous features, or inadequate slide resolution.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Global Academic & Medical Safety Disclaimer */}
      <footer className="disclaimer">
        <ShieldAlert className="disclaimer-icon" size={24} />
        <div className="disclaimer-content">
          <h4>Clinical & Academic Research Disclaimer</h4>
          <p>
            Lung Cancer Detection is an academic deep learning research demonstration designed for clinical decision support and educational analysis. 
            It is <strong>not</strong> an FDA-cleared or CE-marked medical diagnostic device. Neural network inferences must not be used as the 
            sole basis for oncology diagnosis, staging, or treatment planning. All medical findings must be confirmed by a board-certified 
            pathologist or thoracic radiologist in conjunction with comprehensive clinical assessment, immunohistochemical staining, and molecular testing.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
