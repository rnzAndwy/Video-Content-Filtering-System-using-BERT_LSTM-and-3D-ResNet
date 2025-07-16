const OpeningPanel = ({ onGetStarted }) => {
  const [isDialogOpen, setIsDialogOpen] = React.useState(false);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '100vh',
      background: 'linear-gradient(to bottom right, #76b6c4, #235a6b)',
      color: 'white',
      padding: '2rem'
    }}>
      <div style={{maxWidth: '40rem', textAlign: 'center'}}>
        <h1 style={{fontSize: '3rem', fontWeight: 'bold', marginBottom: '1.5rem'}}>VCFS: Video Content Filtering System</h1>
        <p style={{fontSize: '1.25rem', marginBottom: '2rem'}}>Empower your video content with intelligent filtering and analysis. VCFS helps you maintain quality and compliance effortlessly.</p>
        <div style={{display: 'flex', justifyContent: 'center', gap: '1rem'}}>
          <button
            onClick={onGetStarted}
            style={{
              padding: '0.75rem 1.5rem',
              background: 'white',
              color: '#235a6b',
              borderRadius: '9999px',
              fontWeight: '600',
              border: 'none',
              cursor: 'pointer'
            }}
          >
            Get Started
          </button>
          <button
            onClick={() => setIsDialogOpen(true)}
            style={{
              padding: '0.75rem 1.5rem',
              background: 'transparent',
              border: '2px solid white',
              color: 'white',
              borderRadius: '9999px',
              fontWeight: '600',
              cursor: 'pointer'
            }}
          >
            Our Purpose
          </button>
        </div>
      </div>
      {isDialogOpen && (
        <div style={{
          position: 'fixed',
          inset: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <div style={{
            background: '#def3f6',
            color: '#235a6b',
            padding: '2rem',
            borderRadius: '0.5rem',
            maxWidth: '28rem'
          }}>
            <h2 style={{fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1rem'}}>Our Purpose</h2>
            <p style={{marginBottom: '1.5rem'}}>
              VCFS is designed to streamline video content management by automatically filtering and analyzing uploads. Our system helps content creators and managers ensure their videos meet quality standards and comply with platform guidelines, saving time and reducing the risk of policy violations.
            </p>
            <button
              onClick={() => setIsDialogOpen(false)}
              style={{
                padding: '0.5rem 1rem',
                background: '#235a6b',
                color: 'white',
                border: 'none',
                borderRadius: '0.25rem',
                cursor: 'pointer'
              }}
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

// Make the component available globally
window.OpeningPanel = OpeningPanel;