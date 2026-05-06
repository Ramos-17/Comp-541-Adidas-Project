import ForecastChart from "./ForecastChart";
import InventoryHealthTable from "./InventoryHealthTable";
import ClusterView from "./ClusterView";

export default function MainView({ activeView, selectedYear, onYearChange }) {
  if (activeView === "inventory") {
    return <InventoryHealthTable />;
  }

  if (activeView === "clusters") {
    return <ClusterView />;
  }

  return <ForecastChart selectedYear={selectedYear} onYearChange={onYearChange} />;
}
