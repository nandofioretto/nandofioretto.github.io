\ProvidesClass{documentMETADATA}[2018/07/19  v1.9.2 'YAAC: Another Awesome CV' Class]
\LoadClass[letterpaper]{article}
\newif\if@showLinks   \@showLinksfalse

\DeclareOption{green}{
  \def\@green{green}
}

\DeclareOption{red}{
  \def\@red{red}
}

\DeclareOption{indigo}{
  \def\@indigo{indigo}
}
\DeclareOption{orange}{
  \def\@orange{orange}
}

\DeclareOption{monochrome}{
  \def\@monochrome{monochrome}
}

\DeclareOption{localFont}{
  \def\@local{local}
}

\DeclareOption{alternative}{
  \def\@alternative{alternate}
}

\DeclareOption{showLinks}{ \@showLinkstrue}

\ProcessOptions

% Dependences
%A Few Useful Packages
\RequirePackage[english,french]{babel}
\RequirePackage[utf8]{luainputenc}
\RequirePackage{fontspec} 					                 % for loading fonts
\RequirePackage{url,parskip} 	    % other packages for formatting
\RequirePackage[usenames,dvipsnames]{xcolor}
\RequirePackage{fullpage}
\RequirePackage[left=20mm, right=20mm,top=20mm,bottom=20mm]{geometry}
\RequirePackage{fontawesome}
\RequirePackage{hyperref}
\RequirePackage{titlesec}
\RequirePackage{array}
\RequirePackage{enumitem}
\RequirePackage{longtable} 	
\RequirePackage{etoolbox}
\RequirePackage{tikz}
\RequirePackage[skins]{tcolorbox}
\RequirePackage{fancyhdr}
\RequirePackage{xparse}
\RequirePackage{multirow}
\RequirePackage{multicol}
\setlength{\LTleft}{-20pt}

\DeclareUnicodeCharacter{00E9}{\'{e}}
% Define default accent colors
\definecolor{basecolor}{HTML}{000066}       %BLUE

\ifundef{\@green} {
  \ifundef{\@red} {
    \ifundef{\@indigo}{
      \ifundef{\@orange} {
        \ifundef{\@monochrome} {}
        { \definecolor{basecolor}{HTML}{000000}}
      }
      { \definecolor{basecolor}{HTML}{e68a00} }
    }
    { \definecolor{basecolor}{rgb}{0.2, 0.07, 0.48} }
  }
  { \definecolor{basecolor}{HTML}{b30000} }     %RED
}
{ \definecolor{basecolor}{rgb}{0.0, 0.62, 0.38} }

\definecolor{darkGrey}{rgb}{0.48, 0.48, 0.48}

\colorlet{linkcolor}{basecolor}
\colorlet{accentcolor}{linkcolor!90}
\colorlet{symbolcolor}{linkcolor!85}

% Setup hyperref package, and colours for links
%\definecolor{linkcolour}{rgb}{0,0.2,0.6}

\if@showLinks
  \hypersetup{%
    breaklinks
  }%
\else
  \hypersetup{%
    pdfborder = {0 0 0},
    breaklinks
  }%
\fi

% Setup fancyhdr package
\fancyhf{}
\fancyhfoffset{0em}
% Remove head rule
\renewcommand{\headrulewidth}{0pt}
%\fancyfoot[C]{\thepage}
\pagestyle{fancy}

%\pagestyle{empty}                         % non-numbered pages
%\font\fb=''[cmr10]''                      % for use with \LaTeX command

% Setup CV sections
\titleformat{\section}{\Large\raggedright}{}{-12pt}{}[\titlerule]
\titlespacing{\section}{0pt}{0pt}{0pt}
\titlespacing{\subsection}{0pt}{4pt}{0pt}
\titlespacing{\subsubsection}{-20pt}{4pt}{-2pt}

% Configure list
% \setlist[itemize,1]{label=\faAngleRight, nosep, leftmargin=2em} 
% \setlist[itemize,2]{label=\faAngleRight, nosep, leftmargin=1.5em} 
\setlist[itemize,1]{label=\faAngleRight,  leftmargin=1em} 
\setlist[itemize,2]{label=\faAngleRight,  leftmargin=0.5em} 

% Setup Array : new column type
\newcolumntype{R}[1]{>{\hfill}m{#1}}

%Italian hyphenation for the word: ''corporations''
\hyphenation{im-pre-se}

% Setup fonts
% \defaultfontfeatures{Mapping=tex-text}
%

% Define font to use according to localFont option
% \ifundef{\@local}
% {
%   %\setmainfont[BoldFont = Helvetica Neue, ItalicFont=Helvetica Neue Thin Italic ,SmallCapsFont = Helvetica Neue Light]{Helvetica Neue Thin}
%   \setmainfont{Source Sans Pro Light}[
%     BoldFont = SourceSansPro-Regular, 
%     ItalicFont= Source Sans Pro Light Italic]
% }
% {
  \setmainfont{SourceSansPro-Light}[
    Path = fonts/, 
    BoldFont = SourceSansPro-Regular, 
    ItalicFont = SourceSansPro-LightIt]
% }

%New length definition
\newlength{\rightcolumnlength}
\setlength{\rightcolumnlength}{14.6cm}

\newlength{\leftcolumn}
\setlength{\leftcolumn}{2.0cm}

% Font Awesome icons aliases
\newcommand{\mailSymbol}{\faAt}
\newcommand{\locationSymbol}{\faMapMarker}
\newcommand{\infoSymbol}{\faInfo}
\newcommand{\twitterSymbol}{\faTwitter}
\newcommand{\viadeoSymbol}{\faViadeo}
\newcommand{\mobileSymbol}{\faMobilePhone}
\newcommand{\githubSymbol}{\faGithub}
\newcommand{\linkSymbol}{\faHome}

\newcommand\link[2]{{\color{linkcolor}\href{#1}{#2}}}
\newcommand\blink[2]{{\href{#1}{#2}}}
\newcommand\important[1]{\textbf #1}
\newcommand\nemph[1]{{\color{accentcolor}\em{#1}}}
\newcommand{\student}[1]{#1\mbox{\footnotesize\textcolor{symbolcolor}{$^\text{\faUser}$}}}
\newcommand{\venue}[1]{{\scshape \bf{\color{black}{#1}}}}
\renewcommand{\title}[1]{{\bf #1}}


% Create a tag
% Usage: \cvtag{<tag label>}
\newcommand{\cvtag}[1]{%
  \tikz[baseline]\node[anchor=base,draw=darkGrey!70,rounded corners=0.5ex,inner xsep=1ex,inner ysep =0.55ex,text height=1.3ex,text depth=.25ex]{#1};
}

% Render author's name
% Usage: \user{<firstanme>}{<lastname>}
\newcommand\user[2]{\color{accentcolor}{\LARGE #1 \textsc{\textbf{#2}}}\color{Black}}

% Render a text with its symbol
% Usage; \socialtext{<icon>}{<label>}
\newcommand{\socialtext}[2]{\mbox{\textcolor{symbolcolor}{#1}\hspace{0.5em}#2\hspace{1em}}}

% Render a link with its symbol
% Usage; \sociallink{<icon>}{<label>}
\newcommand{\sociallink}[3]{\mbox{\textcolor{symbolcolor}{#1}\hspace{0.5em}\link{#2}{#3}\hspace{1em}}}

% Define author's name
% Usage: \name{<firstname>}{<lastname>}
% Usage: \firstname{<firstname>}
% Usage: \lastname{<lastname>}
% Usage: \familyname{<familyname>}
\newcommand*{\name}[2]{\def\@firstname{#1}\def\@lastname{#2}}
\newcommand*{\firstname}[1]{\def\@firstname{#1}}
\newcommand*{\lastname}[1]{\def\@lastname{#1}}
\newcommand*{\familyname}[1]{\def\@lastname{#1}}
\def\@familyname{\@lastname}

% Define author's tagline
% Usage: \tagline{<tag line>}
\newcommand*{\tagline}[1]{\def\@tagline{#1}}

% Define author's photo
% Usage \photo{<diameter>}{<photo>}
\newcommand{\photo}[2]{\def\@photo{#2}\def\@photodiameter{#1}}

% Render author's address
% Usage: \address{<address>}
\newcommand*{\address}[1]{\socialtext{\locationSymbol}{#1}}

% Render author's infos
% Usage: \infos{<infos>}
\newcommand*{\infos}[1]{\socialtext{\infoSymbol}{#1}}

% Render author's linked-in (optional)
% Usage: \linkedin{<linked-in-nick>}
\newcommand*{\twitter}[1]{\sociallink{\twitterSymbol}{http://www.twitter.com/#1}{#1}}

\newcommand*{\gscholar}[1]{\sociallink{\faGraduationCap}{https://scholar.google.com/citations?hl=en&user=ASf9Q04AAAAJ&view_op=list_works&sortby=pubdate}{#1}}


% Render author's personal website (optional)
% Usage: \personalLink{<website-without-http://www.>}
\newcommand*{\personalLink}[1]{\sociallink{\linkSymbol}{https://www.#1}{#1}}


% Render author's viadeo(optional)
% Usage: \viadeo{<viadeo-nick>}
\newcommand*{\viadeo}[1]{\sociallink{\viadeoSymbol}{http://www.viadeo.com/fr/profile/#1}{viadeo.com/fr/profile/#1}} 

% Render author's github (optional)
% Usage: \github{<github-nick>}
\newcommand*{\github}[1]{\sociallink{\githubSymbol}{https://www.github.com/#1}{github.com/#1}}           % Github icon + URL

% Render author's email (optional)
% Usage: \email{<email adress>}
\newcommand*{\email}[1]{\sociallink{\mailSymbol}{mailto:#1}{#1}}

% Render author's mobile phone (optional)
% Usage: \smartphone{<mobile phone number>}
\newcommand*\smartphone[1]{\socialtext{\mobileSymbol}{#1}}    

% Render author's tagline
\newcommand\resumetitle[1]{
  \ifundef{\@alternative}{
	 \par{
	 	 \bigskip\center{\Large \color{accentcolor}\textbf{#1}\color{Black}}\par
	 }
	 \bigskip
  }{
    \color{accentcolor}\textbf{\large{#1}}\color{black}
  }
}

\newcommand\idphoto{
  \tikz\path[fill overzoom image={\@photo}]rectangle (\linewidth,1.2\linewidth);
  %  \tikz\path[fill overzoom image={\@photo}]circle(radius = 0.5\linewidth);
}

% Define social entries to print in header
\newcommand{\socialinfo}[1]{\def\@socialinfo{#1}}

% Render CV header
% Needs \@firstname, \@lastname and \@tagline to be defined
\newcommand*{\makecvheader}{
  \ifundef{\@alternative}{
    \ifundef{\@photodiameter}{
      \begin{minipage}{\linewidth}
    }{
      \begin{minipage}{\dimexpr\linewidth-\@photodiameter-2em}
    }
    \color{accentcolor}\user{\@firstname}{\@lastname}\color{black}\\
    \small{\@socialinfo}
    \end{minipage}
    \ifdef{\@photodiameter}{
      \begin{minipage}{\@photodiameter}
        \idphoto
      \end{minipage}
    }{}
    \resumetitle{\@tagline}
  }{
    \ifundef{\@photodiameter}{
      \begin{minipage}{\linewidth}
    }{
      \begin{minipage}{\dimexpr\linewidth-\@photodiameter-2em}
    }
    \user{\@firstname}{\@lastname}\\[2pt]
    \resumetitle{\@tagline}
    \smallskip\\
    \small{\@socialinfo}
    \end{minipage}
    \ifdef{\@photodiameter}{
      \begin{minipage}{\@photodiameter}
        \idphoto
      \end{minipage}
    }{}
    %\bigskip
  }
}

% Define a footer for CV
% Usage: \makecvfooter{<left>}{<center>}{<right>}
\newcommand*{\makecvfooter}[3]{%
  \fancyfoot{}
  \fancyfoot[L]{#1}
  \fancyfoot[C]{#2}
  \fancyfoot[R]{#3}
}

% Resume part title definition
\newcommand\sectionTitle[2]{\section{\texorpdfstring{\color{accentcolor}#2\enspace {#1}}{#1}}}
\newcommand\yearTitle[1]{\subsection*{\texorpdfstring{\color{accentcolor}\hspace*{0.75em}\textsc{#1}}{#1}} \vspace{-0.0em}}
\newcommand\subsectionTitle[1]{\subsubsection*{\texorpdfstring{\color{accentcolor}\hspace*{2em}\textsc{#1}}{#1}}\vspace{0.0em}}

% Define the 'keywords' environment
\newenvironment{keywords}{%
  \renewcommand{\arraystretch}{1.1}

  \begin{tabular}{>{}r>{}p{13cm}} 
}{%
  \end{tabular}
}

% Render a scholarshipentry in the scolarship environment
% Usage: \scholarshipentry{<date>}{<description>}
\newcommand\keywordsentry[2]{
  \textbf{#1} &   #2\\ 
}

% Define a new column type for the scholarship environment
\newcolumntype{Y}{>{\raggedleft}p{\leftcolumn}}
\newcolumntype{E}{>{\raggedright\arraybackslash}p{\rightcolumnlength}}
\newcolumntype{T}{>{\raggedleft}p{1.0cm}}

% Define the 'scholarship' environment
\newenvironment{pubs}{%
  %\begin{longtable}{R{\leftcolumn} p{\rightcolumnlength}}
  \begin{longtable}{R{1.0cm} p{15.8cm}}
}{%
  \end{longtable}
  \vspace*{-1em}
}

\newcommand\confentryShort[5]{
  {#1.} & #2. \blink{#5}{\title{#3}}. {#4}.\\
}

\newcommand\confentryShortAwd[6]{
  {#1.} & #2. \blink{#5}{\title{#3}}. {#4}. {\awardText{[#6]}.}\\
}

%%%%%%%%%%%%%%%%%%
%% journals
%%%%%%%%%%%%%%%%%%
\newcommand\journalentry[5]{
  \textbf{[\decvalue{nJ}]} & \blink{#5}{\title{#3}}.~~~[\link{#5}{\footnotesize{\faLink}}] \\
               &  #2. \\ 
               & {#4}.\\[0.5em]
}
\newcommand\journalentryAwd[7]{
  \textbf{[\decvalue{nJ}]} & \blink{#5}{\title{#3}}.~~~[\link{#5}{\footnotesize{\faLink}}]\\
  &  #2. \\ 
  \color{purple}{\faTrophy} & {#4}.\\
  & {\awardText{#6} #7.}\\[0.5em]
}

\newcommand\journalentryIF[6]{
  \textbf{[\decvalue{nJ}]} & \blink{#5}{\title{#3}}.~~~[\link{#5}{\footnotesize{\faLink}}] \\
               &  #2. \\ 
               & {#4}.\\
               & [{\bf Impact Factor}: {#6]}. \\[0.5em]
}
\newcommand\journalentryIFAwd[8]{
  \textbf{[\decvalue{nJ}]} & \blink{#5}{\title{#3}}.~~~[\link{#5}{\footnotesize{\faLink}}]\\
  &  #2. \\ 
  & [{\bf Impact Factor}: {#6]}. \\[0.5em]
  \color{purple}{\faTrophy} & {#4}.\\
  & {\awardText{#7} #8.}\\[0.5em]
}

%%%%%%%%%%%%%%%%%%
%% conferences
%%%%%%%%%%%%%%%%%%
\newcommand\confentry[6]{
  %\textbf{[#1]} 
  \textbf{[\decvalue{nC}]} & \blink{#5}{\title{#3}}.~~~[\link{#5}{\footnotesize{\faLink}}] \\
               &  #2. \\ 
               & {#4}.\\
               & [{\bf Acceptance Rate}: {#6]}. \\[0.5em]
}

\newcommand\confentryAwd[8]{
  %\textbf{[#1]} 
  \textbf{[\decvalue{nC}]} & \blink{#5}{\title{#3}}.~~~[\link{#5}{\footnotesize{\faLink}}]\\
  &  #2. \\ 
  \color{purple}{\faTrophy} & {#4}.\\
  & [{\bf Acceptance Rate}: {#6]}. \\
  & {\awardText{#7} #8.}\\[0.5em]
}

%%%%%%%%%%%%%%%%%%
%% workshops
%%%%%%%%%%%%%%%%%%
\newcommand\wsentry[5]{
  %\textbf{[\label{w#1}{#1}]} 
  \textbf{[\decvalue{nW}]}
  & \blink{#5}{\title{#3}}.~~~[\link{#5}{\footnotesize{\faLink}}] \\
               &  #2. \\ 
               & {#4}.\\[0.5em]
}

\newcommand\wsentryAR[6]{
  %\textbf{[\label{w#1}{#1}]} 
  \textbf{[\decvalue{nW}]}
  & \blink{#5}{\title{#3}}.~~~[\link{#5}{\footnotesize{\faLink}}] \\
               &  #2. \\ 
               & {#4}.\\
               & [{\bf Acceptance Rate}: {#6]}. \\[0.5em]
}

\newcommand\wsentryAwd[7]{
  %\textbf{[\label{w#1}{#1}]} 
  \textbf{[\decvalue{nW}]}
  & \blink{#5}{\title{#3}}.~~~[\link{#5}{\footnotesize{\faLink}}]\\
  &  #2. \\ 
  \color{purple}{\faTrophy} & {#4}.\\
  & {\awardText{#6} #7.}\\[0.5em]
}

\newcommand\wsentryAwdAR[8]{
  %\textbf{[\label{w#1}{#1}]} 
  \textbf{[\decvalue{nW}]}
  & \blink{#5}{\title{#3}}.~~~[\link{#5}{\footnotesize{\faLink}}]\\
  &  #2. \\ 
  \color{purple}{\faTrophy} & {#4}.\\
  & [{\bf Acceptance Rate}: {#6]}. \\
  & {\awardText{#7} #8.}\\[0.5em]
}




\newenvironment{awards}{%
  \begin{longtable}{R{2cm}p{\rightcolumnlength}}
}{%
  \end{longtable}
   \vspace*{-1em}
}

\newcommand*{\press}[2]{
  \link{#1}{\mbox{\textcolor{symbolcolor}{\faExternalLink}}~#2}
}

% Render a award entry in the awards environment
% Usage: \awardentry{<date>}{<description>}{<Event>}{<Link>{press}}
\newcommand\awardentryN[3]{
  #1    &  \textbf{#2},  #3.\\
}

\newcommand\awardentry[5]{
  #1    &  \textbf{#2},  #3.\hspace{0.1em}\press{#4}{#5}\\
}

\newcommand\awardentryD[6]{
  #1    &  \textbf{#2},  #3.\hspace{0.1em}\press{#4}{#5}\\%[1.2em]
        & \begin{minipage}[t]{15.6cm}
          \begin{itemize}
          \item \textcolor{darkGrey}{#6}\vspace{0.4em}
          \end{itemize}
          \end{minipage}\\
}


% Define the 'skills' environment
\newenvironment{skills}{%
  \begin{longtable}{R{\leftcolumn}p{\rightcolumnlength}}
}{%
  \end{longtable}
}

% Render a skill in the skills environment
% Usage: \skill{<skill>}{<level>}
\newcommand\skill[2]{
  \textbf{#1}    & \foreach \x in {1,...,5}{{\color{accentcolor!80} \ifnumgreater{\x}{#2}{\faCircleThin}{\faCircle}}\enspace } \\
}

% Define the 'experiences' environment
\newenvironment{experiences}{%
  \begin{longtable}{R{\leftcolumn}|E}}{%
  \end{longtable} \vspace*{0.0em}}

\newcommand\experience[7]{
  \textbf{#1}    & \textbf{#2, \textsc{#3}, #4}             \\*
  \textbf{#5}    & \begin{minipage}[t]{\rightcolumnlength}
                     #6
                   \end{minipage}               \\*
                 & \footnotesize{\foreach \n in {#7}{\cvtag{\n}}} \\
  }

\newcommand\job[6]{
  \textbf{#2}    & \textbf{#3}, \textit{#4}, #5\\
  \textbf{#1}    & \textsc{#6} \\
  }


\newcommand\teach[4]{
  \textbf{#1}    & \textbf{#2}, \textit{#3}\\*
                 & \textsc{Course Evaluation: } {#4}\\
  }


% Render a consultant experience in the experiences environment
% Usage: 
% \consultantexperience
%  {<End date>}      {<Consultant title>}{<Consulting Enterprise>}{<Country>}
%  {<Start date}     {<Client title>}{<Client business unit>}
%                    {
%                      <Experience description (Could be a list)>
%                    }
%                    {<Technology list>}
\newcommand\consultantexperience[9]{
  \textbf{#1}    & \textbf{#2, \textsc{#3}, #4}   								\\*
  \textbf{#5}    & \emph{#6} pour \textsc{#7}                       				\\*
  				       & 	\begin{minipage}[t]{\rightcolumnlength}
  						        #8
                   	\end{minipage}										\\*
                 & \footnotesize{\foreach \n in {#9}{\cvtag{\n}}} 	\\
  }

\newcommand\emptySeparator{\multicolumn{2}{c}{}\\[-5pt]}

% Define the 'projects' environment
\newenvironment{projects}{%
  \begin{longtable}{l}
}{%
  \end{longtable}
}

\newcommand\grantentrySinglePI[7]{
  \begin{minipage}[t]{\dimexpr(\linewidth) - 2.5em}
    \textbf{\textsc{#1}}
    \cvtag{\$#2} 
    \hfill \textsc{#3--#4}\\
    #5
    \press{#6}{}\\
    \textbf{PI}: #7\\
  \end{minipage}
  \\*
}

\newcommand\grantentryPI[8]{
  \begin{minipage}[t]{\dimexpr(\linewidth) - 2.5em}
    \textbf{\textsc{#1}}
    \cvtag{\$#2}{ of \$#3}
    \hfill \textsc{#4--#5}\\
    #6
    \press{#7}{}\\
    \textbf{PI}: #8\\
  \end{minipage}
  \\*
}

\newcommand\grantentrycoPI[9]{
  \begin{minipage}[t]{\dimexpr(\linewidth) - 2.5em}
    \textbf{\textsc{#1}}
    \cvtag{\$#2}{ of \$#3}
    \hfill \textsc{#4--#5}\\
    #6
    \press{#7}{}\\
    \textbf{PI}: #8, \textbf{co-PI}: #9\\
  \end{minipage}
  \\*
}

% Define the 'project' entry in the 'projects' environment
% Usage:
% \project
%   {<Project name>}{<dates>}
%   {<link1> <link2>}
%   {<Project description>}
%   {<techno1>,<techno2>}
\newcommand\project[5]{
  \begin{minipage}[t]{\dimexpr(\linewidth) - 1.5em}
    \textbf{\textsc{#1}} \hfill \textsc{#2}\smallskip\\
    #3\\
    #4\smallskip\\
    \footnotesize{\foreach \n in {#5}{\cvtag{\n}}}\\
  \end{minipage}
  \\*
}


\newcommand*\twocolumnsection[2]{
  \begin{minipage}[t]{\dimexpr(\linewidth/2) - 3em}
    #1
  \end{minipage}
  \hfill
  \begin{minipage}[t]{\dimexpr(\linewidth/2) - 3em}
    #2
  \end{minipage}
  }

% Define the 'referencees' environment
\newenvironment{referees}{%
}{%
  \hfill
  \bigskip
}

\newcommand*\refereeB[5]{
  \hspace{1em}
  \begin{minipage}[t]{\dimexpr(\linewidth/3) - 3em}
    \begin{tabular}{>{}l >{}l}
      \multicolumn{2}{l}{\textbf{#1}}\\
      \multicolumn{2}{l}{{\emph{#2}}}\\
      \multicolumn{2}{l}{\textsc{#3}}\\
      \quad \mailSymbol & \href{mailto:#4}{#4} \\  
      \quad \faPhone & #5 \\
    \end{tabular}
  \end{minipage}%
}


\newcommand*\referee[5]{
  \hspace{1em}
  \begin{minipage}[t]{\dimexpr(\linewidth/3) - 3em}
    \begin{tabular}{>{}l >{}l}
      \multicolumn{2}{l}{\textbf{#1}}\\
      \multicolumn{2}{l}{\parbox[t]{8cm}{\emph{#2}}}\\
      \multicolumn{2}{l}{\textsc{#3}}\\
      \quad \mailSymbol & \href{mailto:#4}{#4} \\  
      \quad \faPhone & #5 \\
    \end{tabular}
  \end{minipage}%
}

\newcommand*\refereeL[5]{
  \hspace{1em}
  \begin{minipage}[t]{\dimexpr(\linewidth/3) - 3em}
    \begin{tabular}{>{}l >{}l}
      \multicolumn{2}{l}{\textbf{#1}}\\
      \multicolumn{2}{l}{\parbox[t]{7.8cm}{\emph{#2}}}\\[1.3em]
      \multicolumn{2}{l}{\textsc{#3}}\\
      \quad \mailSymbol & \href{mailto:#4}{#4} \\  
      \quad \faPhone & #5 \\
    \end{tabular}
  \end{minipage}%
}

\definecolor{awcol}{rgb}{0.58,0.00,0.70}
\newcommand{\awardText}[1]{\textbf{\textcolor{purple}{#1}}}

\def\procACL{\venue{Association for Computational Linguistics (ACL)}}
\def\procNAACL{\venue{Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL)}}
\def\procWWW{\venue{ACM Web Conferences (WWW)}}
\def\procNeurIPS{\venue{Conference on Neural Information Processing Systems (NeurIPS)}}
\def\procICML{\venue{International Conference on Machine Learning (ICML)}}
\def\procICLR{\venue{International Conference on Learning Representations (ICLR)}}
\def\procAISTATS{\venue{International Conference on Artificial Intelligence and Statistics (AISTATS)}}
\def\procAAMAS{\venue{International Conference on Autonomous Agents and  Multiagent Systems (AAMAS)}}
\def\procAAAI{\venue{AAAI Conference on Artificial Intelligence (AAAI)}}
\def\procIJCAI{\venue{International Joint Conference on Artificial Intelligence (IJCAI)}}
\def\procCP{\venue{International Conference on Principles and Practice of  Constraint Programming (CP)}}
\def\procECAI{\venue{European Conference of Artificial Intelligence (ECAI)}}
\def\procFAccT{\venue{ACM Conference on Fairness, Accountability, and Transparency (FAccT)}}
\def\procISIAM{\venue{International Symposium on Artificial Intelligence and Mathematics (ISAIM)}}
\def\procCPAIOR{\venue{International Conference on the Integration of Constraint Programming, Artificial Intelligence, and Operations Research (CPAIOR)}}
\def\procPADL{\venue{Practical Aspects of Declarative Languages (PADL)}}
\def\procPRIMA{\venue{International Conference on Principles and Practice of Multi-Agent Systems (PRIMA)}}
\def\procPSCC{\venue{Power Systems Computation Conference}}
\def\procCMSB{\venue{Computational Methods in System Biology (CMSB)}}
\def\procPSCC{\venue{Power Systems Computation Conference (PSCC)}}
\def\procECML{\venue{European Conference on Machine Learning (ECML)}}
%\def\procNeurIPS{\venue{Conference on Neural Information Processing Systems (NeurIPS)}}
\def\procUAI{\venue{Conference of Uncertainty on Artificial Intelligence (UAI)}}
\def\JAIR{\venue{Journal of Artificial Intelligence Research (JAIR)}}
\def\AIJ{\venue{Artificial Intelligence Journal (AIJ)}}
\def\TSG{\venue{IEEE Transactions on Smart Grids}}
\def\TPS{\venue{IEEE Transactions on Power Systems}}
\def\procPMAPS{\venue{International Conference on Probabilistic Methods Applied to Power Systems (PMAPS)}}