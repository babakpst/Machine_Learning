(window.webpackJsonp=window.webpackJsonp||[]).push([[127],{"++4Y":function(module,t,e){"use strict";e.r(t),e.d(t,"styles",function(){return R}),e.d(t,"AttemptPage",function(){return z}),e.d(t,"withRedirectToCover",function(){return q});var n=e("VbXa"),i=e.n(n),o=e("q1tI"),a=e.n(o),r=e("MnCE"),u=e("AeFk"),s=e("+LJP"),c=e("8Hdl"),d=e("usGt"),m=e("LHEl"),l=e("3AF4"),p=e("rD/6"),b=e("tcH/"),f=e("s3XC"),v=e("aBK/"),S=e("uAMc"),g=e("RH4a"),O=e("zaiP"),j=e("wIYG"),h=e("kYu0"),y=e("FtgG"),C=e("8ec0"),D=e("l0R+"),I=e("RLfs"),T=e("pJaO"),P=e("kvW3"),k=e("v4A+"),A=e("kIii"),w=e("GZYn"),x=e("pH37"),L=e("VtNW"),M=e.n(L),R={header:function header(t){return Object(u.c)({marginBottom:t.spacing(48)})},points:function points(t){return Object(u.c)({marginTop:t.spacing(8)})},submissionGrade:function submissionGrade(t){return Object(u.c)({marginTop:t.spacing(12)})}},z=function(t){function AttemptPage(){for(var e,n=arguments.length,i=new Array(n),o=0;o<n;o++)i[o]=arguments[o];return(e=t.call.apply(t,[this].concat(i))||this).state={showSkillTaggingDialog:!1},e.handleKeepLearningClick=function(t){Object(x.a)(t)?e.props.redirectToNextItem():e.setState({showSkillTaggingDialog:!0})},e}i()(AttemptPage,t);var e=AttemptPage.prototype;return e.componentDidMount=function componentDidMount(){var t=this.props,e=t.shouldRedirectToCover,n=t.redirectToCover;e&&n()},e.render=function render(){var t=this,e=this.props,n=e.redirectToCover,i=e.redirectToNextItem,o=e.addRedirectToCoverToRouteParams,a=e.examSessionId,r=this.state.showSkillTaggingDialog;return Object(u.d)(O.a,null,function(e){var s=e.item;if(!s)return null;return Object(u.d)(h.a,{standardProctorConfigurationId:s.contentSummary&&"exam"===s.contentSummary.typeName&&s.contentSummary.definition.standardProctorConfigurationId||null},function(e){var d=e.shouldShowTimer,O=e.secondsLeftInLatestAttempt,h=e.refetch,x=e.remainingAttempts;return Object(u.d)(f.b,{onClose:n,backbuttonAriaLabel:M()("Back"),headerLeft:Object(u.d)(l.a,{headerText:s.name,itemTypeText:M()("Graded Quiz"),timeCommitment:s.timeCommitment}),headerRight:Object(u.d)(v.a,{courseId:s.courseId,itemId:s.id,examSessionId:a},function(t){var e=t.isSubmitted;return Object(u.d)(p.a,{deadline:s.deadline,remainingTimeInMs:"number"==typeof O?1e3*O:null,showTimer:d&&!e,timerId:Object(C.a)(s.id)})}),topBanner:Object(u.d)(v.a,{courseId:s.courseId,itemId:s.id,examSessionId:a},function(e){var i=e.isSubmitted,o,a=(s.contentSummary&&"exam"===s.contentSummary.typeName&&s.contentSummary.definition||{}).passingFraction;if(i&&s.itemGrade){var r,c=(s.itemGrade||{}).isPassed,d=Object(w.b)(c,s.isCumulativeGraded,x);return Object(u.d)(m.a,{courseId:s.courseId,itemId:s.id},function(e){var i=e.lockingConfigurationSummary,o=e.bestEvaluation;return Object(u.d)("div",null,Object(u.d)(y.a,{itemGrade:s.itemGrade,computedScore:(o||{}).score,maxScore:(o||{}).maxScore,passingFraction:a,isCumulativeGraded:s.isCumulativeGraded,onKeepLearningClick:function onKeepLearningClick(){return t.handleKeepLearningClick(s.courseId)},remainingAttempts:x,lockingConfigurationSummary:i,onTryAgainClick:n}),Object(T.b)()&&d===w.a.PASSED&&Object(u.d)(T.a,{courseId:s.courseId,itemId:s.id}))})}return null})},Object(u.d)(k.a,{courseId:s.courseId,itemId:s.id},function(t){var e=t.postQuizSuggestions,n=t.loading;return Object(u.d)(v.a,{courseId:s.courseId,itemId:s.id,onQuizSessionQueryCompleted:function onQuizSessionQueryCompleted(){return h()},examSessionId:a},function(t){var i=t.quizFormData,a=t.sessionId,r=t.nextSubmissionDraftId,m=t.attemptScore,l=t.totalPoints,p=t.quizQuestions,f=t.isSubmitted,v=t.hasDraft,O=t.isLimitedFeedback;if(!i||n||!p||!a)return Object(u.d)(j.a,null);if(O)return null;var h=p.map(function(t){return t.prompt.id});return Object(u.d)(g.a,{itemId:s.id,courseId:s.courseId},function(t){var n=t.stepState,p=t.setStepState;return Object(u.d)("div",null,Object(u.d)("div",{css:R.header},Object(u.d)(c.a,{variant:"h1semibold",component:"h2"},s.name),f&&"number"==typeof m?Object(u.d)("div",{css:R.submissionGrade},Object(u.d)(c.a,{variant:"h2semibold",component:"span"},M()("Latest Submission Grade #{grade}%",{grade:Object(D.a)(m)}))):Object(u.d)("div",{css:R.points},Object(u.d)(c.a,{variant:"h4bold",component:"span","aria-label":M()("Total available points #{totalPoints}",{totalPoints:l})},Object(u.d)("span",{"aria-hidden":!0},Object(u.d)(P.b,{message:M()("Total points {totalPoints}"),totalPoints:Object(u.d)(c.a,{variant:"body2",component:"span"},l)}))))),Object(u.d)("div",null,Object(u.d)(A.a,{quizFormData:i,isSubmitted:f,postQuizSuggestions:e,stepState:n})),Object(u.d)(S.a,{ids:h,sessionId:a,nextSubmissionDraftId:r},function(t){var e=t.hasUnfilledResponses,i=t.saveDraft,a=t.autoSaveDraft,r=t.submitDraft,c=t.submitLatestSubmissionDraft;return Object(u.d)(b.a,{hasUnfilledResponses:e,itemId:s.id,courseId:s.courseId,saveDraft:i,autoSaveDraft:a,isSubmitted:f,submitDraft:function submitDraft(){return r?r().then(function(){o()}):Promise.reject()},submitLatestSubmissionDraft:function submitLatestSubmissionDraft(){return c?c().then(function(){o()}):Promise.reject()},hasTimer:d,hasDraft:v,stepState:n,setStepState:p})}))})})}),r&&Object(u.d)(I.b,{hideDialogDuringLoading:!1,scrollLockQuerySelector:".rc-TunnelVisionWrapper__content",dialogTitle:M()("Help us identify the right topics"),dialogDescription:M()("Select the skills that are covered in this quiz"),onClose:i,courseId:s.courseId,itemId:s.id}))})})},AttemptPage}(a.a.Component),q=Object(r.compose)(d.a,Object(s.a)(function(t,e){var n=e.nextItemUrl,i=void 0===n?"":n,o=e.refetchCoverPageData,a=e.refreshProgress,r=function redirectToCover(){o&&o(),t.push({name:"quiz-cover",params:t.params,query:t.location.query}),a()};return{redirectToCover:r,redirectToNextItem:function redirectToNextItem(){i?(t.push(i),a()):r()},addRedirectToCoverToRouteParams:function addRedirectToCoverToRouteParams(){t.push({name:"quiz-attempt",params:t.params,query:{redirectToCover:!0}})},shouldRedirectToCover:t.location.query.redirectToCover}}));t.default=q(z)},PvFy:function(module,t,e){"use strict";var n=e("q1tI"),i=e.n(n),o=e("AeFk"),a=e("8Hdl"),r=e("HDKb"),u=e("5mFy"),s=e("7Dw/"),c=e("/96O"),d={root:function root(t){return Object(o.c)({display:"flex",marginTop:t.spacing(48),marginBottom:t.spacing(48)})},accordionRoot:function accordionRoot(t){return Object(o.c)({marginLeft:"-16px",button:{outline:"none"},padding:t.spacing(0)})},numberCell:Object(o.c)({width:"26px"}),contentCell:Object(o.c)({width:"calc(100% - 26px)"}),cmlContainer:function cmlContainer(t){return Object(o.c)({".rc-CML.cml-cui p":{marginBottom:t.spacing(0)}})}},m=function TextBlock(t){var e=t.textBlock,n=t.isCollapsible,i=void 0!==n&&n,m=t.isExpanded,l=void 0!==m&&m,p=Object(o.d)(a.a,{component:"div",variant:"body1",css:d.cmlContainer},Object(o.d)(c.a,{value:null==e?void 0:e.body}));return Object(o.d)("div",{css:d.root},Object(o.d)("div",{css:d.contentCell},i?Object(o.d)("div",{css:d.accordionRoot},Object(o.d)(r.a,{variant:"silent",defaultExpanded:l},Object(o.d)(u.a,{label:null==e?void 0:e.title}),Object(o.d)(s.a,null,p))):p))};t.a=m},kIii:function(module,t,e){"use strict";var n=e("q1tI"),i=e.n(n),o=e("PvFy"),a=e("9iH2"),r=e("LXPe"),u=e("uFBg"),s=function QuizPartRenderer(t){var e=t.quizFormData,s=t.isSubmitted,c=t.postQuizSuggestions,d=t.stepState,m=Object(r.a)(),l,p=0;return i.a.createElement(n.Fragment,null,e.map(function(t){if(m&&(null==t?void 0:t.typeName)===u.c||(null==t?void 0:t.typeName)===u.a){var e,n,r,b=t;return(null==b?void 0:b.typeName)!==u.a||l||(l=b),i.a.createElement(o.a,{textBlock:null==b?void 0:b.definition,isCollapsible:(null==b?void 0:b.typeName)===u.a,isExpanded:(null===(e=l)||void 0===e?void 0:null===(n=e.definition)||void 0===n?void 0:n.id)===(null==b?void 0:null===(r=b.definition)||void 0===r?void 0:r.id)})}p+=1;var f=m&&(null==t?void 0:t.definition)||t;return i.a.createElement(a.a,{key:null==f?void 0:f.prompt.id,quizQuestion:f,index:p-1,postQuizSuggestions:(s&&c||[]).find(function(t){return t.id===f.prompt.id}),isReadOnly:!!s,isDisabled:!!(null==d?void 0:d.isSaving)||!!(null==d?void 0:d.isSubmitting)})}))};t.a=s},"tcH/":function(module,t,e){"use strict";var n=e("lSNA"),i=e.n(n),o=e("VbXa"),a=e.n(o),r=e("q1tI"),u=e.n(r),s=e("d3Ej"),c=e.n(s),d=e("AeFk"),m=e("HOoY"),l=e("zaiP"),p=e("YmkS"),b=e("qJwm"),f=e("aFpJ"),v=e("CnKM"),S=e("8ec0"),g=e("KvdX"),O=e("rQpo"),j=e("Cqp/"),h=e("ZJgU"),y=e("8xbv"),C=e("PB6g");function ownKeys(t,e){var n=Object.keys(t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(t);e&&(i=i.filter(function(e){return Object.getOwnPropertyDescriptor(t,e).enumerable})),n.push.apply(n,i)}return n}function _objectSpread(t){for(var e=1;e<arguments.length;e++){var n=null!=arguments[e]?arguments[e]:{};e%2?ownKeys(Object(n),!0).forEach(function(e){i()(t,e,n[e])}):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(n)):ownKeys(Object(n)).forEach(function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(n,e))})}return t}var D={root:function root(t){return Object(d.c)({margin:t.spacing(48,0)})},buttonsContainer:function buttonsContainer(t){return Object(d.c)({marginTop:t.spacing(32),paddingBottom:t.spacing(48),borderBottom:"1px solid ".concat(t.palette.gray[300]),marginBottom:t.spacing(12)})},saveButton:function saveButton(t){return Object(d.c)({marginLeft:t.spacing(24)})}},I=function(t){function SubmissionControls(){for(var e,n=arguments.length,i=new Array(n),o=0;o<n;o++)i[o]=arguments[o];return(e=t.call.apply(t,[this].concat(i))||this).state={canSubmit:!1,announcement:""},e.onSubmitClick=function(){var t=e.props,n=t.hasUnfilledResponses,i=t.updateAndSubmitDraft,o=t.showModal,a=function trackedUpdateAndSubmitDraft(){return i&&i().then(function(){m.a.trackComponent({namespace:{app:"open_course",page:"item_page"}},{},"submit","quiz")})};n?o({type:g.a.unansweredQuestions,props:{onPrimaryButtonClick:a}}):i&&(e.setAnnouncement(c()("Submitting assignment")),a())},e.enableSubmit=function(){e.setState({canSubmit:!0})},e.disableSubmit=function(){e.setState({canSubmit:!1})},e.autoSubmit=function(){var t=e.props,n=t.showModal,i=t.hideModal,o=t.submitLatestSubmissionDraft;n({type:g.a.timeExpiredModal,props:{onPrimaryButtonClick:i}}),o&&o()},e.startAutoSubmitTimeout=function(){var t=e.props.expiresAt;"number"==typeof t&&(e.autoSubmitTimeout=window.setTimeout(e.autoSubmit,t-Date.now()))},e.setAnnouncement=function(t){e.setState({announcement:t},function(){setTimeout(function(){e.setState({announcement:""})},150)})},e.updateDraft=function(){var t=e.props.updateDraft;t&&(e.setAnnouncement(c()("Saving draft submission")),t().then(function(){e.setAnnouncement(c()("Draft submission saved successfully"))}))},e}a()(SubmissionControls,t);var e=SubmissionControls.prototype;return e.componentDidMount=function componentDidMount(){var t=this.props,e=t.hasDraft,n=t.autoSaveDraft,i=t.submitLatestSubmissionDraft;!e&&n&&n(),i&&this.startAutoSubmitTimeout()},e.componentDidUpdate=function componentDidUpdate(t){var e=this.props.expiresAt;t.expiresAt!==e&&this.autoSubmitTimeout&&(clearTimeout(this.autoSubmitTimeout),this.autoSubmitTimeout=null,this.startAutoSubmitTimeout())},e.componentWillUnmount=function componentWillUnmount(){this.autoSubmitTimeout&&clearTimeout(this.autoSubmitTimeout)},e.render=function render(){var t=this,e=this.props,n=e.stepState,i=n.isSaving,o=n.isSubmitting,a=n.isAutoSaving,r=e.stepState,u=e.setStepState,s=e.updateDraft,m=e.updateAndSubmitDraft,v=this.state,S=v.canSubmit,g=v.announcement,I=i||a,T=I||o;return Object(d.d)(l.a,null,function(e){var n=e.item;if(n&&n.isPremiumGradingLocked)return Object(d.d)("div",null,Object(d.d)(O.a,{courseId:n.courseId}));return Object(d.d)("div",{css:D.root},Object(d.d)("div",null,Object(d.d)(f.a,{onAgreementComplete:t.enableSubmit,onAgreementIncomplete:t.disableSubmit})),Object(d.d)("div",{css:D.buttonsContainer},m&&Object(d.d)(h.a,{iconPosition:"before",icon:o?Object(d.d)(y.a,null):void 0,onClick:t.onSubmitClick,disabled:T||!S,variant:"primary"},o?c()("Submitting…"):c()("Submit")),s&&Object(d.d)(h.a,{iconPosition:"before",icon:I?Object(d.d)(y.a,null):void 0,onClick:t.updateDraft,disabled:!!T,variant:"secondary",css:D.saveButton},I?c()("Saving..."):c()("Save draft"))),Object(d.d)(p.a,{computedItem:n,itemFeedbackType:b.c.Quiz}),Object(d.d)(C.b,{tagName:"span",role:"region","aria-live":"assertive","aria-atomic":!0},g&&Object(d.d)("span",null,g)),Object(d.d)(j.a,{stepState:r,setStepState:u}))})},SubmissionControls}(u.a.Component),T=function SubmissionControlsContainer(t){var e=t.saveDraft,n=t.submitDraft,i=t.hasTimer,o=t.itemId,a=t.isSubmitted;if(!e||!n)return null;if(!i)return Object(d.d)(g.b,null,function(i){var o=i.showModal,r=i.hideModal;return a?null:Object(d.d)(I,_objectSpread(_objectSpread({},t),{},{updateDraft:e,updateAndSubmitDraft:n,showModal:o,hideModal:r}))});return Object(d.d)(v.a,{id:Object(S.a)(o)},function(i){var o=i.expiresAt;return Object(d.d)(g.b,null,function(i){var r=i.showModal,u=i.hideModal;return a?null:Object(d.d)(I,_objectSpread(_objectSpread({},t),{},{updateDraft:e,updateAndSubmitDraft:n,showModal:r,hideModal:u,expiresAt:o}))})})};t.a=T}}]);
//# sourceMappingURL=127.c9970d8a1065e14d09e4.js.map