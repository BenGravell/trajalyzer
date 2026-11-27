import React from 'react';
import {createRoot} from 'react-dom/client';
import {Theme} from '@astryxdesign/core/theme';
import {themes, defaultThemeSlug} from './astryx-themes';
import App from './App';
import '@astryxdesign/core/reset.css';
import '@astryxdesign/core/astryx.css';
import './global.css';

createRoot(document.getElementById('root')).render(<Theme theme={themes[defaultThemeSlug]}><App /></Theme>);
