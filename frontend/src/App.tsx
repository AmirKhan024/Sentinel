import { Route, Routes } from 'react-router-dom'
import { NavBar } from './components/layout/NavBar'
import { TodayPage } from './pages/TodayPage'
import { OverviewPage } from './pages/OverviewPage'
import { RecommendationsPage } from './pages/RecommendationsPage'
import { EstablishmentDetailPage } from './pages/EstablishmentDetailPage'
import { EstablishmentPlanDetailPage } from './pages/EstablishmentPlanDetailPage'
import { SchedulePage } from './pages/SchedulePage'
import { ScheduleDayPage } from './pages/ScheduleDayPage'
import { BacklogPage } from './pages/BacklogPage'
import { HumanReviewPage } from './pages/HumanReviewPage'
import { GeographicPlanPage } from './pages/GeographicPlanPage'
import { SupervisorPlanReviewPage } from './pages/SupervisorPlanReviewPage'

export function App() {
  return (
    <>
      <NavBar />
      <Routes>
        <Route path="/" element={<TodayPage />} />
        <Route path="/plan" element={<OverviewPage />} />
        <Route path="/recommendations" element={<RecommendationsPage />} />
        <Route path="/establishments/:establishmentId" element={<EstablishmentDetailPage />} />
        <Route
          path="/plan/establishments/:targetInspectionId"
          element={<EstablishmentPlanDetailPage />}
        />
        <Route path="/schedule" element={<SchedulePage />} />
        <Route path="/schedule/day" element={<ScheduleDayPage />} />
        <Route path="/backlog" element={<BacklogPage />} />
        <Route path="/review" element={<HumanReviewPage />} />
        <Route path="/geographic-plan" element={<GeographicPlanPage />} />
        <Route path="/plan-review" element={<SupervisorPlanReviewPage />} />
      </Routes>
    </>
  )
}
